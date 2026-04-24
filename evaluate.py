"""Evaluate segmentation results against ground truth using MoF, F1, and mIoU."""

import argparse

import numpy as np
import torch
from video_dataset import VideoDataset
import metrics as M


def evaluate(data_root, dataset_name, results_path):
    """Load predictions and ground truth, then compute evaluation metrics.

    Args:
        data_root: Root directory containing dataset folders.
        dataset_name: Name of the dataset (e.g., 'desktop_assembly').
        results_path: Path to the .npy file with predicted cluster labels.
    """
    dataset = VideoDataset(
        root_dir=data_root,
        dataset=dataset_name,
        n_frames=None,
        standardise=True,
        split=None,
        random=False,
        n_videos=None,
        action_class=['all']
    )

    results = np.load(results_path, allow_pickle=True)

    pred_batch = []
    gt_batch = []
    mask_batch = []

    for i in range(len(dataset)):
        features, mask, gt, video_fname, unique_actions = dataset[i]
        gt_batch.append(gt)
        pred_batch.append(torch.tensor(results[i]))
        mask_batch.append(torch.ones(len(results[i]), dtype=torch.bool))

    metric_names = ['mof', 'f1', 'miou']
    scores = M.indep_eval_metrics(pred_batch, gt_batch, mask_batch, metrics=metric_names)

    print(f"MoF:  {scores['mof']:.4f}")
    print(f"F1:   {scores['f1']:.4f}")
    print(f"mIoU: {scores['miou']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SmartSeg segmentation results")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing dataset folders")
    parser.add_argument("--dataset", type=str, default="desktop_assembly",
                        help="Dataset name (default: desktop_assembly)")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to the results .npy file")
    args = parser.parse_args()

    evaluate(args.data_root, args.dataset, args.results)
