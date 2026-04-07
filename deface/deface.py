#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
from typing import Dict, Optional, Tuple

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import imageio.plugins.ffmpeg
import cv2
from insightface.app import FaceAnalysis

from deface.centerface import CenterFace

# importlib.metadata is used to fetch the package version without causing a circular import
import importlib.metadata


# ---------------------------------------------------------------------------
# Allowed-face helpers
# ---------------------------------------------------------------------------

def build_face_app() -> FaceAnalysis:
    """Create and prepare an InsightFace FaceAnalysis app."""
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def build_allowed_embeddings(
    allowed_faces_dir: str,
    face_app: FaceAnalysis,
) -> Optional[np.ndarray]:
    """
    Load every image in *allowed_faces_dir*, extract one face embedding per
    image, and return an array of shape (N, 512).  Returns None if the
    directory is empty or no faces are found.
    """
    if not allowed_faces_dir or not os.path.isdir(allowed_faces_dir):
        return None

    embeddings = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = [
        os.path.join(allowed_faces_dir, f)
        for f in os.listdir(allowed_faces_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not paths:
        print(f'[allow-list] No images found in {allowed_faces_dir}')
        return None

    for p in paths:
        img = cv2.imread(p)  # InsightFace expects BGR
        if img is None:
            print(f'[allow-list] Could not read {p}, skipping.')
            continue
        faces = face_app.get(img)
        if not faces:
            print(f'[allow-list] No face detected in {p}, skipping.')
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(face.normed_embedding)
        print(f'[allow-list] Loaded reference face from {os.path.basename(p)}')

    if not embeddings:
        print('[allow-list] Warning: no usable reference faces found.')
        return None

    return np.stack(embeddings)  # (N, 512)


def pad_to_min_size(img: np.ndarray, min_size: int = 320) -> np.ndarray:
    """Pad image with black borders so its smallest side is at least min_size.
    InsightFace's internal detector needs a reasonably large image."""
    h, w = img.shape[:2]
    if h >= min_size and w >= min_size:
        return img
    scale = min_size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def is_allowed_face(
    face_crop: np.ndarray,
    allowed_embeddings: np.ndarray,
    face_app: FaceAnalysis,
    threshold: float = 0.4,
    verbose: bool = False,
) -> bool:
    """
    Return True if *face_crop* (H×W×3 uint8 RGB) matches any allowed face.
    Pads small crops so InsightFace's internal detector can find the face.
    """
    if face_crop.size == 0:
        return False

    # InsightFace expects BGR; frame is RGB (imageio convention)
    bgr_crop = face_crop[:, :, ::-1].copy()

    # Upscale if too small for InsightFace's detector
    bgr_crop = pad_to_min_size(bgr_crop, min_size=320)

    faces = face_app.get(bgr_crop)
    if not faces:
        if verbose:
            print(f'[allow-list] No face found in crop, will blur.')
        return False

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding

    sims = allowed_embeddings @ emb
    best = float(sims.max())
    if verbose:
        print(f'[allow-list] Best similarity: {best:.3f} (threshold: {threshold}) → {"ALLOW" if best > threshold else "BLUR"}')
    return best > threshold


# ---------------------------------------------------------------------------
# Original helpers (unchanged)
# ---------------------------------------------------------------------------

def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg=None,
        mosaicsize: int = 20
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2
        blurred_box = cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            ey, ex = skimage.draw.ellipse(
                (y2 - y1) // 2, (x2 - x1) // 2,
                (y2 - y1) // 2, (x2 - x1) // 2
            )
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:
            frame[y1:y2, x1:x2] = (
                frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255)
                + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
            )
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        pass

    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


# ---------------------------------------------------------------------------
# Core anonymization — now allow-list aware
# ---------------------------------------------------------------------------

def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize,
        # new allow-list params (all optional — backwards compatible)
        allowed_embeddings: Optional[np.ndarray] = None,
        face_app=None,
        allow_threshold: float = 0.4,
        verbose: bool = False,
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)

        # --- allow-list check on unscaled box ---------------------------
        if allowed_embeddings is not None:
            rx1 = max(0, x1)
            ry1 = max(0, y1)
            rx2 = min(frame.shape[1] - 1, x2)
            ry2 = min(frame.shape[0] - 1, y2)
            face_crop = frame[ry1:ry2, rx1:rx2]
            if verbose:
                print(f'[allow-list] Checking face crop size: {face_crop.shape}')
            if is_allowed_face(
                face_crop, allowed_embeddings, face_app, allow_threshold, verbose
            ):
                continue  # skip — this is an allowed face
        # ----------------------------------------------------------------

        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)

        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize
        )


# ---------------------------------------------------------------------------
# Video / image detection (unchanged signatures, new kwargs forwarded)
# ---------------------------------------------------------------------------

def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg=None,
        keep_audio: bool = False,
        mosaicsize: int = 20,
        disable_progress_output=False,
        allowed_embeddings=None,
        face_app=None,
        allow_threshold: float = 0.4,
        verbose: bool = False,
):
    try:
        if 'fps' in ffmpeg_config:
            reader = imageio.get_reader(ipath, fps=ffmpeg_config['fps'])
        else:
            reader = imageio.get_reader(ipath)
        meta = reader.get_meta_data()
        _ = meta['size']
    except Exception:
        if cam:
            print(f'Could not find video device {ipath}. Please set a valid input.')
        else:
            print(f'Could not open file {ipath} as a video file with imageio. Skipping file...')
        return

    if cam:
        nframes = None
        read_iter = cam_read_iter(reader)
    else:
        read_iter = reader.iter_data()
        nframes = reader.count_frames()

    bar = tqdm.tqdm(
        dynamic_ncols=True, total=nframes,
        position=1 if nested else 0,
        leave=True,
        disable=disable_progress_output,
    )

    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        _ffmpeg_config.setdefault('fps', meta['fps'])
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer = imageio.get_writer(opath, format='FFMPEG', mode='I', **_ffmpeg_config)

    for frame in read_iter:
        dets, _ = centerface(frame, threshold=threshold)
        anonymize_frame(
            dets, frame, mask_scale=mask_scale,
            replacewith=replacewith, ellipse=ellipse,
            draw_scores=draw_scores, replaceimg=replaceimg,
            mosaicsize=mosaicsize,
            allowed_embeddings=allowed_embeddings,
            face_app=face_app,
            allow_threshold=allow_threshold,
            verbose=verbose,
        )

        if opath is not None:
            writer.append_data(frame)

        if enable_preview:
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                cv2.destroyAllWindows()
                break
        bar.update()

    reader.close()
    if opath is not None:
        writer.close()
    bar.close()


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg=None,
        mosaicsize: int = 20,
        allowed_embeddings=None,
        face_app=None,
        allow_threshold: float = 0.4,
        verbose: bool = False,
):
    frame = iio.imread(ipath)

    if keep_metadata:
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    dets, _ = centerface(frame, threshold=threshold)
    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse,
        draw_scores=draw_scores, replaceimg=replaceimg,
        mosaicsize=mosaicsize,
        allowed_embeddings=allowed_embeddings,
        face_app=face_app,
        allow_threshold=allow_threshold,
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        imageio.imsave(opath, frame, exif=exif_dict)


# ---------------------------------------------------------------------------
# Misc helpers (unchanged)
# ---------------------------------------------------------------------------

def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg=None):
    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)
    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse,
        draw_scores=draw_scores, replaceimg=replaceimg,
    )
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help='File path(s) or camera device name.')
    parser.add_argument('--output', '-o', default=None, metavar='O')
    parser.add_argument('--thresh', '-t', default=0.2, type=float, metavar='T')
    parser.add_argument('--scale', '-s', default=None, metavar='WxH')
    parser.add_argument('--preview', '-p', default=False, action='store_true')
    parser.add_argument('--boxes', default=False, action='store_true')
    parser.add_argument('--draw-scores', default=False, action='store_true')
    parser.add_argument('--disable-progress-output', default=False, action='store_true')
    parser.add_argument('--mask-scale', default=1.3, type=float, metavar='M')
    parser.add_argument(
        '--replacewith', default='blur',
        choices=['blur', 'solid', 'none', 'img', 'mosaic'])
    parser.add_argument('--replaceimg', default='replace_img.png')
    parser.add_argument('--mosaicsize', default=20, type=int, metavar='width')
    parser.add_argument('--keep-audio', '-k', default=False, action='store_true')
    parser.add_argument('--ffmpeg-config', default={"codec": "libx264"}, type=json.loads)
    parser.add_argument('--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'])
    parser.add_argument('--execution-provider', '--ep', default=None, metavar='EP')
    parser.add_argument('--version', action='version', version=importlib.metadata.version('deface'))
    parser.add_argument('--keep-metadata', '-m', default=False, action='store_true')

    # --- new allow-list arguments ---
    parser.add_argument(
        '--allow-faces', default=None, metavar='DIR',
        help='Path to a folder of reference images for faces that should NOT be blurred.')
    parser.add_argument(
        '--allow-threshold', default=0.4, type=float, metavar='A',
        help='Cosine similarity threshold for the allow-list (higher = stricter). Default: 0.4.')
    parser.add_argument(
        '--verbose', '-v', default=False, action='store_true',
        help='Print allow-list similarity scores for each detected face.')

    parser.add_argument('--help', '-h', action='help')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path, file))
        else:
            ipaths.append(path)

    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    ffmpeg_config = args.ffmpeg_config
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    disable_progress_output = args.disable_progress_output

    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == 'img':
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')

    # --- build allow-list embeddings (if requested) ---
    allowed_embeddings = None
    allowed_embeddings = None
    face_app = None

    if args.allow_faces:
        print('[allow-list] Initialising InsightFace...')
        face_app = build_face_app()
        allowed_embeddings = build_allowed_embeddings(args.allow_faces, face_app)
        if allowed_embeddings is not None:
            print(f'[allow-list] {len(allowed_embeddings)} reference face(s) loaded.')

    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anonymized{ext}'
        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')

        if filetype == 'video' or is_cam:
            video_detect(
                ipath=ipath, opath=opath, centerface=centerface,
                threshold=threshold, cam=is_cam, replacewith=replacewith,
                mask_scale=mask_scale, ellipse=ellipse, draw_scores=draw_scores,
                enable_preview=enable_preview, nested=multi_file,
                keep_audio=keep_audio, ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg, mosaicsize=mosaicsize,
                disable_progress_output=disable_progress_output,
                allowed_embeddings=allowed_embeddings,
                face_app=face_app,
                allow_threshold=args.allow_threshold,
                verbose=args.verbose,
            )
        elif filetype == 'image':
            image_detect(
                ipath=ipath, opath=opath, centerface=centerface,
                threshold=threshold, replacewith=replacewith,
                mask_scale=mask_scale, ellipse=ellipse, draw_scores=draw_scores,
                enable_preview=enable_preview, keep_metadata=keep_metadata,
                replaceimg=replaceimg, mosaicsize=mosaicsize,
                allowed_embeddings=allowed_embeddings,
                face_app=face_app,
                allow_threshold=args.allow_threshold,
                verbose=args.verbose,
            )
        elif filetype is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()
