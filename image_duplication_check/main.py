import argparse
import os
import sys

import cv2
import imgsim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from image_duplication_check import batch_vectorize_images


def is_valid_image(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(10)
            if not (
                header.startswith(b"\xff\xd8")
                or header.startswith(b"\x89PNG\r\n\x1a\n")
                or header[:6] in (b"GIF87a", b"GIF89a")
            ):
                return False
        img = cv2.imread(file_path)
        if img is None:
            return False
        return True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        return False


def convert_windows_path(path: str) -> str:
    if "\\" in path:
        path = path.replace("\\", "/")
        # X:/... -> /mnt/x/...
        path = "/mnt/" + path[0].lower() + path[2:]
    return path


def find_duplicates(
    image_path_list: list[str],
    threshold: float,
    batch_size: int,
    max_workers: int | None = None,
) -> list[dict]:
    vtr = imgsim.Vectorizer()

    images = []
    with tqdm(total=len(image_path_list), desc="Vectorizing") as progress:
        for path, vec in batch_vectorize_images(
            image_path_list, vtr, batch_size=batch_size, max_workers=max_workers
        ):
            images.append((path, vec))
            progress.update()

    if not images:
        return []

    paths = [p for p, _ in images]
    vectors = np.stack([v for _, v in images])

    norms_sq = np.sum(vectors**2, axis=1)
    dot_products = vectors @ vectors.T
    distance_matrix = np.sqrt(
        np.maximum(norms_sq[:, None] - 2 * dot_products + norms_sq[None, :], 0)
    )

    i_indices, j_indices = np.where(
        np.triu(distance_matrix < threshold, k=1)
    )
    result = [
        {"file_path": (paths[i], paths[j]), "dist": distance_matrix[i, j]}
        for i, j in zip(i_indices, j_indices)
    ]

    return result


def show_duplicates(result: list[dict], output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "duplicates")
    os.makedirs(output_dir, exist_ok=True)

    for idx, r in enumerate(result):
        file_path_a = r["file_path"][0]
        file_path_b = r["file_path"][1]
        image_a = cv2.cvtColor(cv2.imread(file_path_a), cv2.COLOR_BGR2RGB)
        image_b = cv2.cvtColor(cv2.imread(file_path_b), cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(image_a)
        ax1.set_title(os.path.basename(file_path_a))
        ax2.imshow(image_b)
        ax2.set_title(os.path.basename(file_path_b))
        fig.suptitle(f"dist={r['dist']:.4f}")

        out_path = os.path.join(output_dir, f"pair_{idx:04d}.png")
        fig.savefig(out_path)
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)
        del image_a, image_b

    print(f"Saved {len(result)} comparison images to {output_dir}")


def delete_duplicates(result: list[dict]) -> None:
    for r in result:
        file_path_b = r["file_path"][1]
        try:
            os.remove(file_path_b)
            print(f"Deleted: {file_path_b}")
        except FileNotFoundError:
            print(f"File not found: {file_path_b}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and remove duplicate images in a directory"
    )
    parser.add_argument("directory", help="Target directory containing images")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Distance threshold for duplicate detection (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="Batch size for vectorization (default: 400)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers for image loading (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete duplicate images without confirmation",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying duplicate image pairs",
    )
    parser.add_argument(
        "--delete-broken",
        action="store_true",
        help="Delete broken images that cannot be loaded",
    )
    args = parser.parse_args()

    dir_path = convert_windows_path(args.directory)

    if not os.path.isdir(dir_path):
        print(f"Error: '{dir_path}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    file_path_list = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

    image_path_list = []
    broken_files = []
    for file_path in file_path_list:
        if is_valid_image(file_path):
            image_path_list.append(file_path)
        else:
            broken_files.append(file_path)
            print(f"Skipping invalid file: {file_path}", file=sys.stderr)

    if broken_files:
        print(f"Found {len(broken_files)} broken/invalid files.")
        if args.delete_broken:
            for f in broken_files:
                try:
                    os.remove(f)
                    print(f"Deleted broken file: {f}")
                except OSError as e:
                    print(f"Failed to delete {f}: {e}", file=sys.stderr)

    if not image_path_list:
        print("No valid images found.")
        return

    print(f"Found {len(image_path_list)} valid images.")

    result = find_duplicates(
        image_path_list, args.threshold, args.batch_size, args.max_workers
    )
    total_combinations = len(image_path_list) * (len(image_path_list) - 1) // 2
    print(
        f"Found {len(result)} duplicate pairs (out of {total_combinations} combinations)"
    )

    if not result:
        return

    if not args.no_show:
        show_duplicates(result)

    if args.delete:
        delete_duplicates(result)
    else:
        answer = input("Delete duplicate images? (y/n): ")
        if answer.lower() == "y":
            delete_duplicates(result)


if __name__ == "__main__":
    main()
