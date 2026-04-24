import argparse
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import tsm_network

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Feature extraction parameters
STRIDE = 6
FPS = 30
BATCH_SIZE = 16

# TSM network parameters
ENCODER_HIDDEN = 1024
CHANNEL_NUM = 4
DECODER_HIDDEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_decay(distance, sigma=100):
    """Apply Gaussian decay based on temporal distance."""
    return np.exp(-distance ** 2 / (2 * sigma ** 2))


def tsm_matrix(feature, nk=23):
    """Generate temporal self-similarity matrix and perform clustering.

    Args:
        feature: Input feature array of shape [N, D].
        nk: Target number of clusters for hierarchical merging.

    Returns:
        Cluster labels for each frame, shape [N].
    """
    features_t = torch.tensor(feature).to(device)
    N = features_t.shape[0]

    model = tsm_network.SJNET(
        frame_size=N,
        encoder_hidden=ENCODER_HIDDEN,
        channel_num=CHANNEL_NUM,
        decoder_hidden=DECODER_HIDDEN,
    )
    tsm = model.get_tsm(features_t)

    if len(tsm.shape) == 4:
        tsm = tsm[0, :, :, 0]
    elif len(tsm.shape) == 3:
        tsm = tsm[:, :, 0]

    # Scale TSM and apply temporal Gaussian decay
    scaler = MinMaxScaler()
    tsm = scaler.fit_transform(tsm)
    n = tsm.shape[0]
    for i in range(n):
        for j in range(n):
            distance = abs(i - j)
            tsm[i, j] *= gaussian_decay(distance)

    # Spectral clustering with silhouette-based model selection
    k_values = range(nk + 1, 40)
    best_k = None
    best_silhouette = -1
    best_labels = None

    for k in k_values:
        spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = spectral.fit_predict(tsm)
        score = silhouette_score(tsm, labels)

        if score > best_silhouette:
            best_silhouette = score
            best_k = k
            best_labels = labels

    # Reorder labels by order of first appearance
    unique_ordered = []
    for label in best_labels:
        if label not in unique_ordered:
            unique_ordered.append(label)

    label_map = {old: new for new, old in enumerate(unique_ordered)}
    reordered_labels = np.array([label_map[label] for label in best_labels])

    # Hierarchical clustering to merge into exactly nk clusters
    centroids = np.array([tsm[reordered_labels == i].mean(axis=0) for i in range(best_k)])
    Z = linkage(centroids, method='ward')
    hc_labels = fcluster(Z, nk, criterion='maxclust')

    # Reorder hierarchical labels by first appearance
    unique_ordered = []
    for label in hc_labels:
        if label not in unique_ordered:
            unique_ordered.append(label)

    label_map = {old: new for new, old in enumerate(unique_ordered)}
    final_labels = np.array([label_map[label] for label in hc_labels])

    # Map spectral cluster assignments to final hierarchical labels
    mapping = {i: final_labels[i] for i in range(len(final_labels))}
    final_cluster_labels = np.array([mapping[cluster] for cluster in reordered_labels])

    assert len(final_cluster_labels) == N, (
        f"Label count mismatch: expected {N}, got {len(final_cluster_labels)}"
    )
    return final_cluster_labels


def process_and_save(folder_path, output_path, nk=23):
    """Process all .npy feature files in a folder and save clustering results.

    Args:
        folder_path: Directory containing .npy feature files.
        output_path: Path to save the output .npy results file.
        nk: Target number of clusters.
    """
    feature_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.npy')
    ])

    results = []
    for feature_file in tqdm(feature_files, desc="Processing files"):
        feature = np.load(feature_file)
        pred = tsm_matrix(feature, nk)
        results.append(pred)

    np.save(output_path, np.array(results, dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSeg temporal segmentation pipeline")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing .npy feature files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for results .npy file")
    parser.add_argument("--nk", type=int, default=23,
                        help="Target number of clusters (default: 23)")
    args = parser.parse_args()

    process_and_save(args.features_dir, args.output, nk=args.nk)
