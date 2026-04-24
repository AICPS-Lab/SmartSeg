# SmartSeg

A framework for robust temporal segmentation of wearable-camera (egocentric) videos.

## Overview

SmartSeg addresses the unique challenges of egocentric footage that conventional video segmentation methods struggle with:

- **Unstable viewpoints** from continuous head and body movements
- **Diverse activities** spanning multiple environments
- **Variable duration** ranging from sub-minute clips to sessions exceeding 20 minutes

The framework uses an encoding-driven pipeline that combines **Temporal Self-Similarity Matrix (TSM)** encoding with **positional encoding** to denoise long, unstable videos and accurately detect event boundaries. A **nonparametric clustering** stage then merges fine-grained segments into semantically meaningful units.

## Architecture

```
Input Features (.npy)
        |
   [CustomEncoder]          (tsm_network.py)
   Multi-scale temporal branches:
     - Short-range (1x1 conv)
     - Mid-range (3x3 conv + dilated conv)
     - Long-range (MLP-Mixer)
        |
   [TSM Computation]        (smartseg.py)
   Cosine similarity + Gaussian decay
        |
   [Spectral Clustering]
   Silhouette-based model selection
        |
   [Hierarchical Merging]
   Ward linkage to target cluster count
        |
   Segment Labels
```

## Installation

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib tqdm einops torchmetrics
```

## Usage

### Run Segmentation

```bash
python smartseg.py \
    --features_dir /path/to/features/ \
    --output /path/to/results.npy \
    --nk 23
```

**Arguments:**
| Argument | Description | Default |
|---|---|---|
| `--features_dir` | Directory containing `.npy` feature files | *required* |
| `--output` | Output path for results `.npy` file | *required* |
| `--nk` | Target number of clusters | `23` |

### Evaluate Results

```bash
python evaluate.py \
    --data_root /path/to/data/ \
    --dataset desktop_assembly \
    --results /path/to/results.npy
```

**Arguments:**
| Argument | Description | Default |
|---|---|---|
| `--data_root` | Root directory containing dataset folders | *required* |
| `--dataset` | Dataset name | `desktop_assembly` |
| `--results` | Path to the results `.npy` file | *required* |

## Project Structure

```
SmartSeg/
├── smartseg.py       # Main segmentation pipeline
├── tsm_network.py    # TSM neural network (SJNET + CustomEncoder)
├── mlp_mixer.py      # MLP-Mixer block for long-range temporal modeling
├── metrics.py        # Evaluation metrics (MoF, F1, mIoU)
├── evaluate.py       # Evaluation script for comparing predictions to ground truth
├── utils.py          # Visualization and feature standardization utilities
├── LICENSE           # MIT License
└── README.md
```


## Contact

For questions about the code or paper, please reach out to:
- [Yilin Liu](https://scholar.google.com/citations?user=oHCMDqcAAAAJ&hl=en)
- [Hanchen David Wang](https://scholar.google.com/citations?user=ZAi6VHsAAAAJ&hl=en)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{liu2026smartseg,
  title={SmartSeg: A non-parametric approach for wearable camera video temporal segmentation},
  author={Liu, Yilin and Wang, Hanchen David and Fu, Haowei and Mason, Madison Lee and Li, Fanjie and Wise, Alyssa and Levin, Daniel T and Biswas, Gautam and Ma, Meiyi},
  journal={Pervasive and Mobile Computing},
  pages={102223},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
