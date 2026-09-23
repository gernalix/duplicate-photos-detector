from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .engine import DEFAULT_DB, DuplicatePhotoEngine, MatchThresholds
from .vision import (
    format_local_timestamp,
    render_feature_diagnostic,
    render_side_by_side,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Local duplicate/reverse-image matcher. It does not identify people.",
)
console = Console()


def _engine(db: Path, phash_same: int = 8, geometry_min_inliers: int = 10):
    return DuplicatePhotoEngine(
        db,
        MatchThresholds(
            phash_same=phash_same,
            geometry_min_inliers=geometry_min_inliers,
        ),
    )


def _print_results(results) -> None:
    table = Table(show_lines=True)
    for name in (
        "rank", "class", "confidence", "first seen", "pHash", "crop",
        "SIFT/RANSAC", "perspective", "embedding", "path",
    ):
        table.add_column(name)

    for idx, result in enumerate(results, 1):
        crop_avg = (
            f"{result.crop_average_distance:.1f}"
            if math.isfinite(result.crop_average_distance)
            else "—"
        )
        embed = (
            f"{result.embedding_similarity:.3f}"
            if result.embedding_similarity is not None
            else "—"
        )
        table.add_row(
            str(idx),
            result.classification,
            result.confidence,
            format_local_timestamp(result.first_seen_ts),
            str(result.phash_distance),
            f"{result.crop_regions} regions / {crop_avg}",
            f"{result.geometry.inliers}/{result.geometry.good_matches} "
            f"({result.geometry.inlier_ratio:.2f})",
            f"{result.geometry.perspective_strength:.4f}",
            embed,
            result.path,
        )
    console.print(table)


@app.command("index")
def index_command(
    archive: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB,
    embeddings: Annotated[bool, typer.Option("--embeddings")] = False,
    prune: Annotated[bool, typer.Option("--prune")] = False,
):
    """Incrementally index new or changed archive images."""
    stats = DuplicatePhotoEngine(db).index_archive(
        archive,
        embeddings=embeddings,
        prune=prune,
    )
    console.print(
        f"scanned={stats.scanned} indexed={stats.indexed} skipped={stats.skipped} "
        f"failed={stats.failed} pruned={stats.pruned}"
    )
    for failure in stats.failures:
        console.print(f"[red]{failure}[/red]")


@app.command("find")
def find_command(
    query: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB,
    top: Annotated[int, typer.Option("--top", min=1, max=100)] = 10,
    embeddings: Annotated[bool, typer.Option("--embeddings")] = False,
    diagnostics: Annotated[Path | None, typer.Option("--diagnostics")] = None,
    phash_same: Annotated[int, typer.Option("--phash-same", min=0, max=64)] = 8,
    geometry_min_inliers: Annotated[int, typer.Option("--geometry-min-inliers", min=4)] = 10,
):
    """Find exact, transformed, cropped or screen-photographed copies."""
    engine = _engine(db, phash_same, geometry_min_inliers)
    results = engine.search(query, top=top, embeddings=embeddings)
    if not results:
        console.print("Index is empty. Run 'grindr-photo index <archive>' first.")
        raise typer.Exit(1)

    _print_results(results)

    timeline = engine.strong_timeline(results)
    if timeline:
        console.print("\nStrong-match timeline:")
        for ts in timeline:
            console.print(f"  {format_local_timestamp(ts)}")
    else:
        console.print("\nNo strong same-image match found.")

    if diagnostics is not None:
        diagnostics.mkdir(parents=True, exist_ok=True)
        written = 0
        for rank, result in enumerate(results[:3], 1):
            if result.classification == "NO_MATCH":
                continue
            prefix = diagnostics / f"rank-{rank:02d}"
            render_side_by_side(
                query,
                result.path,
                prefix.with_name(prefix.name + "-comparison.jpg"),
                label=f"{result.classification} {result.confidence}",
            )
            render_feature_diagnostic(
                query,
                result.path,
                prefix.with_name(prefix.name + "-features.jpg"),
            )
            written += 1
        console.print(f"Diagnostics written for {written} result(s): {diagnostics}")


@app.command("similar")
def similar_command(
    query: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB,
    top: Annotated[int, typer.Option("--top", min=1, max=100)] = 10,
    embeddings: Annotated[bool, typer.Option("--embeddings")] = False,
):
    """Show closest archive candidates, including weak/no-match candidates."""
    results = DuplicatePhotoEngine(db).search(
        query,
        top=top,
        embeddings=embeddings,
    )
    if not results:
        console.print("Index is empty.")
        raise typer.Exit(1)
    _print_results(results)


@app.command("watch")
def watch_command(
    archive: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB,
    interval: Annotated[float, typer.Option("--interval", min=2.0)] = 10.0,
    embeddings: Annotated[bool, typer.Option("--embeddings")] = False,
    prune: Annotated[bool, typer.Option("--prune/--no-prune")] = True,
):
    """Continuously synchronize the local index with an archive directory."""
    engine = DuplicatePhotoEngine(db)
    console.print(
        f"Watching {archive.resolve()} every {interval:g}s; database={Path(db).expanduser()}"
    )
    try:
        while True:
            stats = engine.index_archive(
                archive,
                embeddings=embeddings,
                prune=prune,
            )
            if stats.indexed or stats.failed or stats.pruned:
                console.print(
                    f"indexed={stats.indexed} failed={stats.failed} "
                    f"pruned={stats.pruned} skipped={stats.skipped}"
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("Stopped.")


if __name__ == "__main__":
    app()
