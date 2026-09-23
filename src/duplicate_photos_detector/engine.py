from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .db import PhotoIndex
from .embeddings import OpenClipEmbedder, nearest_embeddings, vector_to_blob
from .vision import (
    GeometryStats,
    crop_hash_metrics,
    geometric_match,
    infer_first_seen,
    is_supported_image,
    load_rgb,
    perceptual_hashes,
    phash_distance,
    sha256_file,
)

XDG_DATA_HOME = Path(
    os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
).expanduser()
DEFAULT_DB = XDG_DATA_HOME / "duplicate-photos-detector" / "index.sqlite3"


@dataclass(slots=True)
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    pruned: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MatchThresholds:
    phash_same: int = 8
    crop_region_cutoff: int = 1
    crop_bit_error_rate: float = 0.25
    geometry_min_good: int = 12
    geometry_min_inliers: int = 10
    geometry_min_ratio: float = 0.35
    screen_perspective: float = 0.015
    embedding_similar: float = 0.90
    hash_candidates: int = 80
    geometry_candidates: int = 30


@dataclass(slots=True)
class MatchResult:
    image_id: int
    path: str
    first_seen_ts: float
    classification: str
    confidence: str
    score: float
    sha_equal: bool
    phash_distance: int
    crop_regions: int
    crop_average_distance: float
    geometry: GeometryStats
    embedding_similarity: float | None = None


class DuplicatePhotoEngine:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB,
        thresholds: MatchThresholds | None = None,
    ):
        self.db = PhotoIndex(db_path)
        self.thresholds = thresholds or MatchThresholds()

    def index_archive(
        self,
        root: Path | str,
        *,
        embeddings: bool = False,
        prune: bool = False,
    ) -> IndexStats:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Archive directory does not exist: {root}")

        stats = IndexStats()
        embedder = OpenClipEmbedder() if embeddings else None
        existing_paths: list[str] = []

        for path in sorted(root.rglob("*")):
            if not is_supported_image(path):
                continue
            stats.scanned += 1
            resolved = path.resolve()
            existing_paths.append(str(resolved))

            try:
                st = resolved.stat()
                previous = self.db.get_by_path(resolved)
                embedding_is_ready = (
                    not embeddings
                    or (
                        previous is not None
                        and previous["embedding"] is not None
                        and previous["embedding_model"] == embedder.model_key
                    )
                )

                if (
                    previous is not None
                    and previous["size_bytes"] == st.st_size
                    and previous["mtime_ns"] == st.st_mtime_ns
                    and embedding_is_ready
                ):
                    stats.skipped += 1
                    continue

                image = load_rgb(resolved)
                phash, crop_hash = perceptual_hashes(image)
                vector = embedder.encode(image) if embedder else None

                self.db.upsert(
                    {
                        "path": str(resolved),
                        "archive_root": str(root),
                        "sha256": sha256_file(resolved),
                        "phash": phash,
                        "crop_hash": crop_hash,
                        "width": image.width,
                        "height": image.height,
                        "size_bytes": st.st_size,
                        "mtime_ns": st.st_mtime_ns,
                        "first_seen_ts": infer_first_seen(resolved, image),
                        "indexed_ts": time.time(),
                        "embedding": vector_to_blob(vector) if vector is not None else None,
                        "embedding_dim": int(vector.shape[0]) if vector is not None else None,
                        "embedding_model": embedder.model_key if embedder else None,
                    }
                )
                stats.indexed += 1
            except Exception as exc:
                stats.failed += 1
                stats.failures.append(f"{resolved}: {exc}")

        if prune:
            stats.pruned = self.db.prune_root(root, existing_paths)
        return stats

    def _classify(
        self,
        *,
        sha_equal: bool,
        phash_dist: int,
        crop_regions: int,
        crop_avg: float,
        geometry: GeometryStats,
        embedding_similarity: float | None,
    ) -> tuple[str, float]:
        t = self.thresholds

        if sha_equal:
            return "EXACT", 1.0

        crop_good = (
            crop_regions >= t.crop_region_cutoff
            and crop_avg <= 64 * t.crop_bit_error_rate
        )
        geometry_good = (
            geometry.good_matches >= t.geometry_min_good
            and geometry.inliers >= t.geometry_min_inliers
            and geometry.inlier_ratio >= t.geometry_min_ratio
        )
        phash_good = phash_dist <= t.phash_same

        phash_score = max(0.0, 1.0 - phash_dist / 32.0)

        crop_score = 0.0
        if crop_regions:
            crop_score = min(
                0.96,
                0.62
                + min(crop_regions, 5) * 0.055
                - min(crop_avg, 64.0) / 256.0,
            )

        geometry_score = 0.0
        if geometry.good_matches:
            geometry_score = min(
                0.98,
                0.52
                + min(geometry.inliers, 30) / 100.0
                + min(geometry.inlier_ratio, 1.0) * 0.16,
            )

        if (
            geometry_good
            and geometry.perspective_strength >= t.screen_perspective
        ):
            return "SCREEN_CAPTURE", max(0.78, geometry_score, crop_score)

        if phash_good or crop_good or geometry_good:
            return "SAME_IMAGE", max(phash_score, crop_score, geometry_score)

        if (
            embedding_similarity is not None
            and embedding_similarity >= t.embedding_similar
        ):
            return "VISUALLY_SIMILAR", embedding_similarity

        return "NO_MATCH", max(
            phash_score * 0.6,
            crop_score * 0.6,
            geometry_score * 0.6,
            (embedding_similarity or 0.0) * 0.7,
        )

    @staticmethod
    def _confidence(classification: str, score: float) -> str:
        if classification == "NO_MATCH":
            return "none"
        if score >= 0.90:
            return "very-high"
        if score >= 0.78:
            return "high"
        if score >= 0.65:
            return "medium"
        return "low"

    def search(
        self,
        query_path: Path | str,
        *,
        top: int = 10,
        embeddings: bool = False,
    ) -> list[MatchResult]:
        query_path = Path(query_path).expanduser().resolve()
        if not query_path.is_file():
            raise ValueError(f"Query image does not exist: {query_path}")

        query_image = load_rgb(query_path)
        query_sha = sha256_file(query_path)
        query_phash, query_crop = perceptual_hashes(query_image)
        rows = self.db.all_rows()
        if not rows:
            return []

        by_id = {int(row["id"]): row for row in rows}
        exact_ids = {
            int(row["id"])
            for row in rows
            if row["sha256"] == query_sha
        }

        phash_distances = {
            int(row["id"]): phash_distance(query_phash, row["phash"])
            for row in rows
        }
        hash_order = sorted(
            phash_distances,
            key=lambda image_id: phash_distances[image_id],
        )
        candidate_ids = (
            set(hash_order[: self.thresholds.hash_candidates])
            | exact_ids
        )

        embedding_scores: dict[int, float] = {}
        if embeddings:
            embedder = OpenClipEmbedder()
            query_vector = embedder.encode(query_image)
            embedding_rows = [
                (
                    int(row["id"]),
                    row["embedding"],
                    int(row["embedding_dim"]),
                )
                for row in rows
                if row["embedding"] is not None
                and row["embedding_dim"] is not None
                and row["embedding_model"] == embedder.model_key
            ]
            for image_id, score in nearest_embeddings(
                query_vector,
                embedding_rows,
                max(top * 5, 50),
            ):
                embedding_scores[image_id] = score
                candidate_ids.add(image_id)

        crop_metrics: dict[int, tuple[int, float]] = {}
        prelim: list[tuple[int, int, int, float]] = []

        for image_id in candidate_ids:
            row = by_id[image_id]
            try:
                regions, avg = crop_hash_metrics(
                    query_crop,
                    row["crop_hash"],
                    bit_error_rate=self.thresholds.crop_bit_error_rate,
                )
            except Exception:
                regions, avg = 0, float("inf")

            crop_metrics[image_id] = (regions, avg)
            prelim.append(
                (
                    image_id,
                    phash_distances[image_id],
                    -regions,
                    avg,
                )
            )

        prelim.sort(key=lambda item: (item[1], item[2], item[3]))
        geometry_ids = {
            item[0]
            for item in prelim[: self.thresholds.geometry_candidates]
        } | exact_ids

        geometry: dict[int, GeometryStats] = {}
        for image_id in geometry_ids:
            geometry[image_id] = geometric_match(
                query_path,
                by_id[image_id]["path"],
            )

        results: list[MatchResult] = []
        for image_id in candidate_ids:
            row = by_id[image_id]
            regions, crop_avg = crop_metrics.get(
                image_id,
                (0, float("inf")),
            )
            geom = geometry.get(image_id, GeometryStats())
            emb = embedding_scores.get(image_id)

            classification, score = self._classify(
                sha_equal=image_id in exact_ids,
                phash_dist=phash_distances[image_id],
                crop_regions=regions,
                crop_avg=crop_avg,
                geometry=geom,
                embedding_similarity=emb,
            )

            results.append(
                MatchResult(
                    image_id=image_id,
                    path=row["path"],
                    first_seen_ts=float(row["first_seen_ts"]),
                    classification=classification,
                    confidence=self._confidence(classification, score),
                    score=float(score),
                    sha_equal=image_id in exact_ids,
                    phash_distance=phash_distances[image_id],
                    crop_regions=regions,
                    crop_average_distance=crop_avg,
                    geometry=geom,
                    embedding_similarity=emb,
                )
            )

        priority = {
            "EXACT": 0,
            "SCREEN_CAPTURE": 1,
            "SAME_IMAGE": 2,
            "VISUALLY_SIMILAR": 3,
            "NO_MATCH": 4,
        }
        results.sort(
            key=lambda result: (
                priority[result.classification],
                -result.score,
                result.phash_distance,
                result.first_seen_ts,
            )
        )
        return results[:top]

    @staticmethod
    def strong_timeline(results: list[MatchResult]) -> list[float]:
        accepted = {"EXACT", "SAME_IMAGE", "SCREEN_CAPTURE"}
        return sorted(
            {
                result.first_seen_ts
                for result in results
                if result.classification in accepted
            }
        )
