from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"

    @property
    def key(self) -> str:
        return f"{self.model_name}:{self.pretrained}"


class OpenClipEmbedder:
    def __init__(self, config: EmbeddingConfig | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Embedding support is not installed. Run: pip install -e '.[embed]'"
            ) from exc

        self.config = config or EmbeddingConfig()
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.device,
        )
        self.model.eval()

    @property
    def model_key(self) -> str:
        return self.config.key

    def encode(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            vector = self.model.encode_image(tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector[0].detach().cpu().numpy().astype(np.float32)


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim).copy()


def nearest_embeddings(
    query: np.ndarray,
    rows: list[tuple[int, bytes, int]],
    top_k: int,
) -> list[tuple[int, float]]:
    if not rows:
        return []

    ids = [row[0] for row in rows]
    matrix = np.vstack(
        [blob_to_vector(row[1], row[2]) for row in rows]
    ).astype(np.float32)
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)

    if len(rows) >= 1000:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            scores, positions = index.search(query, min(top_k, len(rows)))
            return [
                (ids[int(pos)], float(score))
                for score, pos in zip(scores[0], positions[0])
                if pos >= 0
            ]
        except ImportError:
            pass

    scores = matrix @ query[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [(ids[int(i)], float(scores[int(i)])) for i in order]
