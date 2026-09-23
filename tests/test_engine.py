from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from duplicate_photos_detector.engine import DuplicatePhotoEngine
from duplicate_photos_detector.vision import (
    crop_hash_metrics,
    geometric_match,
    load_rgb,
    perceptual_hashes,
    phash_distance,
    sha256_file,
)


def make_feature_rich(path: Path, size=(640, 480)) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for x in range(30, size[0] - 30, 55):
        draw.line((x, 20, size[0] - x // 3, size[1] - 20), fill="black", width=4)
    for y in range(40, size[1] - 40, 60):
        draw.rectangle((30, y, 180 + y // 3, y + 35), outline="navy", width=4)
    draw.ellipse((250, 120, 430, 300), outline="red", width=8)
    draw.text((220, 330), "DUPLICATE PHOTO TEST 2026", fill="black")
    image.save(path, quality=96)


def test_sha256_exact(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    make_feature_rich(a)
    b.write_bytes(a.read_bytes())
    assert sha256_file(a) == sha256_file(b)


def test_phash_survives_resize_and_reencode(tmp_path: Path):
    original = tmp_path / "original.jpg"
    resized = tmp_path / "resized.jpg"
    make_feature_rich(original)
    image = load_rgb(original).resize((480, 360))
    image.save(resized, quality=70)
    h1, _ = perceptual_hashes(load_rgb(original))
    h2, _ = perceptual_hashes(load_rgb(resized))
    assert phash_distance(h1, h2) <= 8


def test_crop_hash_produces_matching_region(tmp_path: Path):
    original = tmp_path / "original.jpg"
    cropped = tmp_path / "cropped.jpg"
    make_feature_rich(original)
    load_rgb(original).crop((80, 50, 580, 430)).save(cropped, quality=88)
    _, c1 = perceptual_hashes(load_rgb(original))
    _, c2 = perceptual_hashes(load_rgb(cropped))
    regions, _ = crop_hash_metrics(c1, c2)
    assert regions >= 1


def test_geometric_match_handles_perspective(tmp_path: Path):
    original = tmp_path / "original.jpg"
    warped = tmp_path / "warped.jpg"
    make_feature_rich(original)
    source = cv2.imread(str(original))
    h, w = source.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[45, 25], [w - 70, 5], [w - 20, h - 45], [20, h - 5]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(source, matrix, (w, h))
    assert cv2.imwrite(str(warped), out)

    stats = geometric_match(warped, original)
    assert stats.homography_found
    assert stats.inliers >= 8
    assert stats.inlier_ratio >= 0.25


def test_index_and_search_exact(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    original = archive / "original.jpg"
    query = tmp_path / "query.jpg"
    make_feature_rich(original)
    query.write_bytes(original.read_bytes())

    engine = DuplicatePhotoEngine(tmp_path / "index.sqlite3")
    stats = engine.index_archive(archive)
    assert stats.indexed == 1

    results = engine.search(query, top=3)
    assert results
    assert results[0].classification == "EXACT"
    assert Path(results[0].path) == original.resolve()
