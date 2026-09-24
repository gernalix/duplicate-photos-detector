# duplicate-photos-detector

Local reverse-image matcher for a personal photo archive. It is designed for cases where the same underlying photo may reappear later after recompression, resizing, cropping, or being photographed from a screen.

The tool does **not** identify people and does **not** perform face recognition. It answers the narrower question: “Have I already seen this same underlying image, or something visually similar?”

## Matching pipeline

1. **SHA-256** — exact byte-for-byte duplicates.
2. **pHash** — recompressed/resized/lightly edited copies.
3. **Crop-resistant hash** — cropped variants using ImageHash multi-hashes.
4. **SIFT + RANSAC homography** — geometric verification for perspective changes, including photos of a display.
5. **Optional OpenCLIP embedding** — local fallback for visually similar images. This is not treated as proof of identity.
6. **Optional FAISS** — accelerates embedding nearest-neighbour search for larger archives.

Results are classified as:

- `EXACT`: same file bytes.
- `SAME_IMAGE`: strong evidence for the same underlying photograph after ordinary transformations.
- `SCREEN_CAPTURE`: strong geometric match plus perspective evidence consistent with photographing/reprojecting the source image.
- `VISUALLY_SIMILAR`: semantic/visual similarity only; manual confirmation required.
- `NO_MATCH`: no sufficiently strong evidence.

## Privacy

Everything is local. The SQLite index stores paths, hashes, timestamps, dimensions and optional embedding vectors. Original photos stay in the archive directory and are never copied into the repository or uploaded by this program.

OpenCLIP weights may need to be downloaded once by the upstream library. After the weights are cached, matching itself is local.

## Install

Python 3.11+ is recommended.

```bash
git clone https://github.com/gernalix/duplicate-photos-detector.git
cd duplicate-photos-detector
python3 -m pip install --user --break-system-packages .
```

Optional embedding support:

```bash
python3 -m pip install --user --break-system-packages '.[embed]'
```

Optional FAISS acceleration:

```bash
python3 -m pip install --user --break-system-packages '.[embed,faiss]'
```

## Commands

The package installs both `grindr-photo` and the generic alias `duplicate-photo`.

### Index an archive

```bash
grindr-photo index ~/Pictures/GrindrPhotos
```

The default database is:

```text
~/.local/share/duplicate-photos-detector/index.sqlite3
```

Indexing is incremental. Unchanged files are skipped.

To calculate OpenCLIP embeddings while indexing:

```bash
grindr-photo index ~/Pictures/GrindrPhotos --embeddings
```

### Find whether a query image already exists

```bash
grindr-photo find ~/Downloads/query.jpg
```

The report includes:

- match class and confidence;
- archive path;
- earliest known appearance;
- pHash distance;
- crop-resistant matching regions;
- SIFT/RANSAC inliers and inlier ratio;
- perspective strength;
- optional embedding similarity;
- a compact timeline of strong matching occurrences.

### Generate visual diagnostics

```bash
grindr-photo find ~/Downloads/query.jpg --diagnostics ~/tmp/photo-match
```

For the strongest results this writes:

- a side-by-side query/archive comparison;
- a SIFT feature-match image showing geometrically verified correspondences.

This is intended for manual validation of borderline matches.

### Group every strong occurrence

```bash
grindr-photo group ~/Downloads/photo.jpg
```

`group` broadens the hash/crop scan across the whole index, applies geometric verification to plausible candidates, and prints every strong `EXACT`, `SAME_IMAGE` or `SCREEN_CAPTURE` occurrence it finds, together with the occurrence count, earliest appearance and timeline. This is the command to use when the main question is “when have I already seen this underlying photo?”

### Ask only for nearest visual candidates

```bash
grindr-photo similar ~/Downloads/query.jpg --top 10
```

With an embedding-enabled index:

```bash
grindr-photo similar ~/Downloads/query.jpg --top 10 --embeddings
```

### Keep the index updated

For archives that change only occasionally, run incremental indexing periodically instead of keeping a polling process resident:

```bash
grindr-photo index ~/Pictures/GrindrPhotos --prune
```

The Fedora templates below run that command every 5 minutes. Unchanged images are skipped, while `--prune` removes index rows for files that were deleted or moved.

The interactive `watch` command is still available when near-immediate updates are temporarily useful:

```bash
grindr-photo watch ~/Pictures/GrindrPhotos --interval 10
```

`watch` recursively scans the archive on every interval, so it is not the recommended always-on mode for a mostly static archive. If another importer owns creation of archive files, invoking `grindr-photo index ... --prune` once after a successful import is even cheaper than periodic polling.

## First-seen timestamps

For each file the index tries, in order:

1. EXIF `DateTimeOriginal` when present;
2. filesystem modification time;
3. current indexing time.

When a path is re-indexed, an earlier already-known first-seen value is preserved. Search results can therefore tell you approximately when an image first entered the archive even if the current query was received much later.

These timestamps are evidence from the local archive, not timestamps from Grindr and not a link to a Grindr conversation.

## Screen photographs

A photo of a monitor/phone display often defeats exact hashes and can weaken perceptual hashes because of perspective, borders, glare, moiré and resampling. SIFT feature correspondences followed by a RANSAC homography are therefore used as a geometric verification stage.

`SCREEN_CAPTURE` is intentionally conservative: it requires a strong geometric match and measurable projective distortion. A front-on display photo may instead be classified as `SAME_IMAGE`, which is still the desired answer.

## Thresholds

Defaults are deliberately conservative and live in `MatchThresholds` in `engine.py`. They should be calibrated against representative real archive examples before treating confidence labels as stable. The CLI exposes the most useful pHash and geometry limits.

No threshold should be interpreted as proof that two different photographs depict the same person.

## Tests

```bash
pytest
```

The automated suite covers exact duplicates, re-encoding/resizing, crop-resistant hashing, database indexing and geometric verification on synthetic transformed images.

## Fedora indexing

The repository includes user-systemd templates:

- `systemd/duplicate-photos-detector-index.service.template`: one incremental indexing pass with pruning.
- `systemd/duplicate-photos-detector-index.timer.template`: starts the service after login and then every 5 minutes.
- `systemd/duplicate-photos-detector-watch.service.template`: always-on archive watcher with systemd restart.

Copy the chosen unit or timer pair into `~/.config/systemd/user/` without the `.template` suffix and replace `__ARCHIVE_DIR__` and `__DB_PATH__`. For the always-on watcher:

```bash
systemctl --user daemon-reload
systemctl --user enable --now duplicate-photos-detector-watch.service
```

Run either the watcher or the periodic timer for one archive, not both. Machine-specific paths and activation remain local to Fedora.

For this project, the canonical roadmap contains the local-Fedora completion task so installation, model caching, real-photo calibration and service activation can be executed where the archive actually exists.
