from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageOps

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}


@dataclass(slots=True)
class GeometryStats:
    keypoints_query: int = 0
    keypoints_candidate: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    perspective_strength: float = 0.0
    homography_found: bool = False


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_rgb(path: Path | str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    phash = imagehash.phash(image, hash_size=8)
    crop_hash = imagehash.crop_resistant_hash(
        image,
        min_segment_size=500,
        segmentation_image_size=300,
    )
    return str(phash), str(crop_hash)


def phash_distance(left: str, right: str) -> int:
    return imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)


def crop_hash_metrics(
    left: str,
    right: str,
    *,
    bit_error_rate: float = 0.25,
) -> tuple[int, float]:
    a = imagehash.hex_to_multihash(left)
    b = imagehash.hex_to_multihash(right)
    matches, total_distance = a.hash_diff(b, bit_error_rate=bit_error_rate)
    average = float(total_distance) / matches if matches else math.inf
    return matches, average


def _parse_exif_datetime(image: Image.Image) -> float | None:
    try:
        exif = image.getexif()
        values = []
        for tag in (36867, 36868, 306):
            value = exif.get(tag)
            if value:
                values.append(value)
        try:
            exif_ifd = exif.get_ifd(34665)
            for tag in (36867, 36868):
                value = exif_ifd.get(tag)
                if value:
                    values.insert(0, value)
        except Exception:
            pass

        for raw in values:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "ignore")
            try:
                return datetime.strptime(
                    str(raw).strip(), "%Y:%m:%d %H:%M:%S"
                ).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return None


def infer_first_seen(path: Path, image: Image.Image) -> float:
    exif_ts = _parse_exif_datetime(image)
    if exif_ts is not None:
        return exif_ts
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


def format_local_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _read_gray(path: Path | str, max_dim: int = 1600) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _feature_match_data(query_path: Path | str, candidate_path: Path | str):
    q = _read_gray(query_path)
    c = _read_gray(candidate_path)
    if q is None or c is None:
        return q, c, [], [], [], None, None

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(q, None)
    kp2, des2 = sift.detectAndCompute(c, None)
    kp1 = kp1 or []
    kp2 = kp2 or []
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return q, c, kp1, kp2, [], None, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for pair in matcher.knnMatch(des1, des2, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    if len(good) < 4:
        return q, c, kp1, kp2, good, None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return q, c, kp1, kp2, good, homography, mask


def geometric_match(query_path: Path | str, candidate_path: Path | str) -> GeometryStats:
    q, _, kp1, kp2, good, homography, mask = _feature_match_data(
        query_path, candidate_path
    )
    stats = GeometryStats(
        keypoints_query=len(kp1),
        keypoints_candidate=len(kp2),
        good_matches=len(good),
    )
    if homography is None or mask is None or q is None:
        return stats

    stats.inliers = int(mask.ravel().sum())
    stats.inlier_ratio = stats.inliers / len(good) if good else 0.0
    stats.homography_found = True

    h = homography.astype(float)
    if abs(h[2, 2]) > 1e-12:
        h /= h[2, 2]
    height, width = q.shape[:2]
    stats.perspective_strength = max(
        abs(float(h[2, 0])) * width,
        abs(float(h[2, 1])) * height,
    )
    return stats


def render_side_by_side(
    query_path: Path | str,
    candidate_path: Path | str,
    output_path: Path | str,
    *,
    label: str = "",
) -> None:
    q = load_rgb(query_path)
    c = load_rgb(candidate_path)
    q.thumbnail((700, 700), Image.Resampling.LANCZOS)
    c.thumbnail((700, 700), Image.Resampling.LANCZOS)
    margin = 24
    header = 48
    canvas = Image.new(
        "RGB",
        (q.width + c.width + margin * 3, max(q.height, c.height) + header + margin * 2),
        "white",
    )
    canvas.paste(q, (margin, header + margin))
    canvas.paste(c, (q.width + margin * 2, header + margin))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), f"QUERY    {label}", fill="black")
    draw.text((q.width + margin * 2, 10), "ARCHIVE MATCH", fill="black")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def render_feature_diagnostic(
    query_path: Path | str,
    candidate_path: Path | str,
    output_path: Path | str,
) -> bool:
    q, c, kp1, kp2, good, _, mask = _feature_match_data(query_path, candidate_path)
    if q is None or c is None or not good:
        return False

    selected = good
    if mask is not None:
        selected = [m for m, keep in zip(good, mask.ravel().astype(bool)) if keep]
    selected = selected[:80]
    if not selected:
        return False

    drawn = cv2.drawMatches(
        q,
        kp1,
        c,
        kp2,
        selected,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), drawn))
