import imgsim
import cv2
import numpy as np
import os
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from typing import Generator


def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)

def vectorize_image(image: np.ndarray, vtr=imgsim.Vectorizer()) -> np.ndarray:
    return vtr.vectorize(image)

def load_and_vectorize_image(image_path: str, vtr=imgsim.Vectorizer()) -> np.ndarray:
    return vectorize_image(load_image(image_path), vtr)

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
            futures = [
                (path, executor.submit(load_and_vectorize_image, path, vtr))
                for path in batch
            ]
            for path, future in futures:
                yield (path, future.result())
