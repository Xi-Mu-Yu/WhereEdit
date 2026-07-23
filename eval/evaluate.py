"""Standalone PIE-Bench evaluation (copied from PnPInversion)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from matrics_calculator import MetricsCalculator

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
PIE_ROOT = ROOT / "sd-turbo" / "pie_bench"

DEFAULT_MAPPING = PIE_ROOT / "mapping_file.json"
DEFAULT_SRC_FOLDER = PIE_ROOT / "annotation_images"
DEFAULT_RESULT = EVAL_ROOT / "evaluation_result.csv"

FULL_METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]

all_tgt_image_folders = {
    "1_ddim+p2p": str(PIE_ROOT / "output" / "WhereEdit" / "annotation_images"),
}


def mask_decode(encoded_mask, image_shape=(512, 512)):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def calculate_metric(
    metrics_calculator,
    metric,
    src_image,
    tgt_image,
    src_mask,
    tgt_mask,
    src_prompt,
    tgt_prompt,
):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "structure_distance_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_structure_distance(
            src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
        )
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_structure_distance(
            src_image, tgt_image, src_mask, tgt_mask
        )
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    raise ValueError(f"Unknown metric: {metric}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIE-Bench editing results.")
    parser.add_argument(
        "--annotation-mapping-file",
        type=str,
        default=str(DEFAULT_MAPPING),
        help="PIE mapping JSON path.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=list(FULL_METRICS),
    )
    parser.add_argument(
        "--src-image-folder",
        type=str,
        default=str(DEFAULT_SRC_FOLDER),
        help="Original PIE annotation image root.",
    )
    parser.add_argument(
        "--tgt-methods",
        nargs="+",
        type=str,
        default=["1"],
        help="Method key prefix filter (e.g. '1' selects 1_ddim+p2p when --evaluate-whole-table).",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default=str(DEFAULT_RESULT),
        help="Output CSV path.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--edit-category-list",
        nargs="+",
        type=str,
        default=[str(i) for i in range(10)],
    )
    parser.add_argument(
        "--evaluate-whole-table",
        action="store_true",
        default=True,
        help="Evaluate all configured methods whose key prefix matches --tgt-methods.",
    )
    parser.add_argument(
        "--no-evaluate-whole-table",
        dest="evaluate_whole_table",
        action="store_false",
    )
    parser.add_argument(
        "--tgt-image-folder",
        type=str,
        default=None,
        help="Override edited image root for method 1_ddim+p2p.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tgt_folders = dict(all_tgt_image_folders)
    if args.tgt_image_folder is not None:
        tgt_folders["1_ddim+p2p"] = args.tgt_image_folder

    tgt_image_folders = {}
    if args.evaluate_whole_table:
        for key, folder in tgt_folders.items():
            if key[0] in args.tgt_methods:
                tgt_image_folders[key] = folder
    else:
        for key in args.tgt_methods:
            tgt_image_folders[key] = tgt_folders[key]

    metrics_calculator = MetricsCalculator(args.device)
    result_path = args.result_path

    with open(result_path, "w", newline="") as handle:
        csv_write = csv.writer(handle)
        csv_head = []
        for tgt_image_folder_key in tgt_image_folders:
            for metric in args.metrics:
                csv_head.append(f"{tgt_image_folder_key}|{metric}")
        csv_write.writerow(["file_id"] + csv_head)

    with open(args.annotation_mapping_file, "r", encoding="utf-8") as handle:
        annotation_file = json.load(handle)

    for key, item in annotation_file.items():
        if item["editing_type_id"] not in args.edit_category_list:
            continue

        print(f"evaluating image {key} ...")
        base_image_path = item["image_path"]
        mask = mask_decode(item["mask"])
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        mask = mask[:, :, np.newaxis].repeat([3], axis=2)

        src_image_path = os.path.join(args.src_image_folder, base_image_path)
        src_image = Image.open(src_image_path)

        evaluation_result = [key]
        for tgt_image_folder_key, tgt_image_folder in tgt_image_folders.items():
            tgt_image_path = os.path.join(tgt_image_folder, base_image_path)
            print(f"evluating method: {tgt_image_folder_key}")
            tgt_image = Image.open(tgt_image_path)
            if tgt_image.size[0] != tgt_image.size[1]:
                tgt_image = tgt_image.crop(
                    (tgt_image.size[0] - 512, tgt_image.size[1] - 512, tgt_image.size[0], tgt_image.size[1])
                )

            for metric in args.metrics:
                print(f"evluating metric: {metric}")
                evaluation_result.append(
                    calculate_metric(
                        metrics_calculator,
                        metric,
                        src_image,
                        tgt_image,
                        mask,
                        mask,
                        original_prompt,
                        editing_prompt,
                    )
                )

        with open(result_path, "a+", newline="") as handle:
            csv.writer(handle).writerow(evaluation_result)

    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
