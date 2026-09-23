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
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional embedding support:

```bash
pip install -e '.[embed]'
```

Optional FAISS acceleration:

```bash
pip install -e '.[embed,faiss]'
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

### Ask only for nearest visual candidates

```bash
grindr-photo similar ~/Downloads/query.jpg --top 10
```

With an embedding-enabled index:

```bash
grindr-photo similar ~/Downloads/query.jpg --top 10 --embeddings
```

### Keep the index updated

```bash
grindr-photo watch ~/Pictures/GrindrPhotos
```

`watch` performs cheap incremental scans at a configurable interval and indexes only new or changed files:

```bash
grindr-photo watch ~/Pictures/GrindrPhotos --interval 10
```

It is suitable for a Fedora user-level systemd service. The repository includes a service template; machine-specific installation/enabling should be done locally.

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

## Fedora service

`systemd/duplicate-photos-detector-watch.service.template` is a template, not an enabled unit. Copy it into `~/.config/systemd/user/`, replace the placeholders with the local checkout/archive paths, then run `systemctl --user daemon-reload && systemctl --user enable --now ...`.

For this project, the canonical roadmap contains the local-Fedora completion task so installation, model caching, real-photo calibration and service activation can be executed where the archive actually exists.
