# SmartSeg
SmartSeg is a framework for robust temporal segmentation of wearable-camera videos, designed to address the unique challenges of egocentric footage that conventional video segmentation methods struggle with. Unlike videos from fixed cameras, wearable camera recordings suffer from unstable viewpoints due to continuous head and body movements, capture diverse activities across multiple environments, and vary widely in duration, from sub-minute clips to sessions exceeding 20 minutes. SmartSeg addresses these challenges through an encoding-driven pipeline that combines temporal self-similarity matrix (TSM) encoding with positional encoding to denoise long, unstable videos and accurately detect event boundaries, followed by a nonparametric clustering stage that merges fine-grained segments into semantically meaningful units.
This repository contains the official implementation of SmartSeg along with scripts for training, evaluation, and reproducing the results reported in our paper. Across public benchmarks, SmartSeg consistently outperforms state-of-the-art baselines, achieving 67.9% accuracy on a mixed egocentric/third-person dataset and 61.6% on a fully egocentric dataset where prior methods failed to generalize. We also include a case study on a real-world nursing simulation video, demonstrating SmartSeg's ability to segment noisy, long-duration recordings containing over a dozen distinct clinical activities. See the sections below for installation instructions, dataset preparation, and usage examples.

## Contact

For questions about the code or paper, please reach out to:
- [Yilin Liu](https://scholar.google.com/citations?user=oHCMDqcAAAAJ&hl=en)
- [Hanchen David Wang](https://scholar.google.com/citations?user=ZAi6VHsAAAAJ&hl=en)

## Citation

If you find this work useful in your research, please consider citing:

    @article{liu2026smartseg,
      title={SmartSeg: A non-parametric approach for wearable camera video temporal segmentation},
      author={Liu, Yilin and Wang, Hanchen David and Fu, Haowei and Mason, Madison Lee and Li, Fanjie and Wise, Alyssa and Levin, Daniel T and Biswas, Gautam and Ma, Meiyi},
      journal={Pervasive and Mobile Computing},
      pages={102223},
      year={2026},
      publisher={Elsevier}
    }
