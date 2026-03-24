import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

import cv2
import imgsim
import numpy as np
import torch


def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)


def vectorize_image(image: np.ndarray, vtr: imgsim.Vectorizer) -> np.ndarray:
    with torch.no_grad():
        return vtr.vectorize(image)


def load_and_vectorize_image(image_path: str, vtr: imgsim.Vectorizer) -> np.ndarray:
    image = load_image(image_path)
    vec = vectorize_image(image, vtr)
    del image
    return vec


def image_distance(image_a_vec: np.ndarray, image_b_vec: np.ndarray) -> float:
    return imgsim.distance(image_a_vec, image_b_vec)


def batch_vectorize_images(
    image_path_list: list[str],
    vtr: imgsim.Vectorizer,
    batch_size: int = 100,
    max_workers: int | None = None,
) -> Generator[tuple[str, np.ndarray], None, None]:
    """画像をバッチ単位でベクトル化し、1枚ずつyieldする。

    バッチごとにThreadPoolExecutorで並列読み込み・ベクトル化を行い、
    結果をyieldした後にバッチ分のメモリを解放する。
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    for i in range(0, len(image_path_list), batch_size):
        batch = image_path_list[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_and_vectorize_image, path, vtr): path
                for path in batch
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                yield (path, future.result())
                del future
        gc.collect()
