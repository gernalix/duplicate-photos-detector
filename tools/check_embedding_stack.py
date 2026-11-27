"""Use synthetic images and vectors to verify the installed embedding path."""
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory

import faiss
import numpy as np
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from duplicate_photos_detector.cli import app
from duplicate_photos_detector.embeddings import nearest_embeddings, vector_to_blob


def main():
    assert sys.prefix == sys.base_prefix
    with TemporaryDirectory(prefix="embedding-fixture-") as temporary:
        root = Path(temporary)
        image = Image.new("RGB", (160, 160), (16, 32, 64))
        shape = ImageDraw.Draw(image)
        shape.rectangle((8, 20, 120, 98), fill=(144, 190, 30))
        shape.ellipse((28, 69, 144, 143), fill=(162, 30, 215))
        archive = root / "archive"
        archive.mkdir()
        image.save(archive / "fixture.png")
        path = root / "fixture.sqlite3"
        result = CliRunner().invoke(app, ["index", str(archive), "--db", str(path), "--embeddings"])
        if result.exception is not None:
            raise result.exception
        assert result.exit_code == 0 and "indexed=1" in result.output, result.output
        with sqlite3.connect(path) as database:
            blob, dim, model = database.execute(
                "SELECT embedding, embedding_dim, embedding_model FROM images"
            ).fetchone()
        vector = np.frombuffer(blob, dtype=np.float32)
        assert dim == len(vector) == 512 and np.isfinite(vector).all()
        assert np.isclose(np.linalg.norm(vector), 1)
        assert model == "ViT-B-32:laion2b_s34b_b79k"

    query = np.array([1.0, 0.0], dtype=np.float32)
    ordinary = vector_to_blob(np.array([0.0, 1.0], dtype=np.float32))
    rows = [(number, ordinary, 2) for number in range(999)] + [(999, vector_to_blob(query), 2)]
    assert nearest_embeddings(query, rows, 1) == [(999, 1.0)]
    assert faiss.IndexFlatIP(2).d == 2
    names = ("torch", "torchvision", "open-clip-torch", "faiss-cpu")
    print(json.dumps({"versions": {name: version(name) for name in names},
                      "dimensions": dim, "model": model, "faiss_rows": len(rows), "base_python": True}))


if __name__ == "__main__":
    main()
