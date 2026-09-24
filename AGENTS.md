# Codex instructions — duplicate-photos-detector

Keep the project local-first and narrow.

- Match the same underlying image across encoding, crop and perspective changes.
- Do not perform face recognition, biometric identification, or infer that two different photos depict the same person.
- Never commit, upload, attach or otherwise exfiltrate private photos, SQLite indexes, embeddings, diagnostics or cached model weights.
- Machine-specific archive paths belong in local configuration/systemd units, not committed source.
- Preserve the staged matcher: SHA-256 -> perceptual/crop hashes -> SIFT/RANSAC -> optional OpenCLIP.
- Semantic embeddings are fallback evidence only.
- Prefer targeted tests; do not add unrelated image-management features.
- For a mostly static archive, prefer one-shot incremental indexing from a 5-minute user-systemd timer (or an importer post-success hook) over an always-on polling service.
- If thresholds are tuned from real local photos, commit only aggregate threshold/test results, never private filenames or image contents.
