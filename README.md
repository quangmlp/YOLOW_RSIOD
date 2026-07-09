# YOLOW-RSIOD: YOLO-World-Based Continual Learning for Remote Sensing Incremental Object Detection

## 🌟 Introduction

**YOLOW-RSIOD** is an advanced framework for **Remote Sensing Incremental Object Detection (RSIOD)**, built upon the strong open-vocabulary recognition capability of [YOLO-World](https://github.com/AILab-CVC/YOLO-World).

When learning newly introduced object categories in continual learning scenarios, conventional object detectors often suffer from **catastrophic forgetting**, where performance on previously learned classes significantly degrades. **YOLOW-RSIOD** addresses this challenge by introducing:

* **Multimodal Knowledge Distillation**: A comprehensive cross-modal distillation strategy that combines classification distillation and bounding-box regression distillation to align visual and linguistic representations.
* **Robust Pseudo-Labeling**: A pseudo-label generation mechanism that leverages the strong zero-shot capability of YOLO-World to produce reliable pseudo-labels for old classes without requiring access to original images from previous training tasks.

This repository focuses on challenging remote sensing scenarios and provides complete configurations for the **DIOR** and **DOTA** datasets.

---

## 📂 Repository Structure

* `yolo_world/`: Core implementation of YOLO-IOD, including multimodal backbones, a customized PAFPN neck, cross-distillation loss functions, and data preprocessors.
* `third_party/mmyolo/`: A customized version of the MMYolo framework used as the underlying detection ecosystem.
* `configs/`: Complete configuration files for incremental learning scenarios, such as DIOR `10+10` and DOTA `5+5+5`.
* `tools/`: Training and evaluation scripts, including `train.py` and `test.py`.
* `script/`: Utility scripts for pseudo-label generation, COCO-format dataset conversion, and dataset splitting.
* `Colab_Notebooks/`: A collection of Google Colab-ready Jupyter notebooks for visualizing data distributions, detection results, and failure-case analyses.
* `assets/`: Diagrams and visual resources.

---

## 🛠️ Installation and Environment Setup

We recommend using Anaconda for environment management.

### 1. Create a Conda Environment

```bash
conda create -n yoloiod python=3.10 -y
conda activate yoloiod
```

### 2. Install Dependencies

```bash
pip install setuptools==69.5.1
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim

mim install "mmengine>=0.10.3"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

pip install -r requirements/basic_requirements.txt
pip install "numpy<2" transformers==4.30.2 scikit-learn prettytable albumentations
```

### 3. Install MMYolo and YOLO-IOD

Install the local `mmyolo` package and the main project:

```bash
# Build and install MMYolo
cd third_party/mmyolo 
pip install -v -e . --no-build-isolation

# Build and install YOLO-IOD
cd ../..
pip install -v -e . --no-build-isolation
```

---

## 📊 Data Preparation

Download the **DIOR** and **DOTA** datasets, and convert their annotations into the standard COCO JSON format.

The datasets should be organized under the `data/` directory as follows:

```text
data/
├── DIOR/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── annotation/
│       └── annotation/
│           ├── train_task_1.json
│           ├── test_task_1.json
│           ├── ...
└── DOTA/
    ├── images/
    │   ├── train/
    │   └── test/
    └── annotation/
        └── annotation/
            ├── train_task_0.json
            ├── test_task_0.json
            ├── ...
```

*Note: Ensure that the dataset paths are consistent with the corresponding configuration files. Please refer to the `script/` directory for annotation conversion utilities.*

---

## 🚀 Training and Evaluation

YOLOW-RSIOD is trained incrementally across multiple learning stages, referred to as tasks.

For example, in the **DIOR 10+10** setting, the model is first trained on 10 base classes in Task 0 and is then sequentially trained on 10 newly introduced classes in Task 1.

### DIOR 10 + 10

```bash
# Stage 1: Train Task 0 on base classes
python tools/train.py configs/dior_10_10/yolo_iod_dior_10_10_task0.py

# Stage 2: Train Task 1 on newly introduced incremental classes
python tools/train.py configs/dior_10_10/yolo_iod_dior_10_10_task1.py
```

### DOTA 5 + 5 + 5

```bash
# Stage 1: Train Task 0 on base classes
python tools/train.py configs/dota_5_5_5/yolo_iod_dota_5_5_5_task0.py

# Stage 2: Train Task 1 on newly introduced incremental classes
python tools/train.py configs/dota_5_5_5/yolo_iod_dota_5_5_5_task1.py

# Stage 3: Train Task 2 on the next set of incremental classes
python tools/train.py configs/dota_5_5_5/yolo_iod_dota_5_5_5_task2.py
```

### Testing

To evaluate a saved checkpoint for a specific task, run:

```bash
python tools/test.py \
    configs/dior_10_10/yolo_iod_dior_10_10_task1.py \
    work_dirs/yolo_iod_dior_10_10_task1/epoch_20.pth
```

---

## 🔬 Visualization and Inference with Jupyter Notebooks

The `Colab_Notebooks/` directory contains self-contained notebooks designed for interactive visualization. These notebooks are configured to run efficiently in Google Colab environments:

1. **`Notebook_1_DataStats.ipynb`**: Analyzes class distributions and object-size statistics on the DIOR and DOTA datasets.
2. **`Notebook_2_Detection_Results.ipynb`**: Performs inference and visualizes side-by-side detection results using the `MMDet` visualization utilities.
3. **`Notebook_3_Failure_Cases.ipynb`**: Identifies and analyzes representative detection failure cases, including *misclassification*, *missing objects*, and *false detections*.
4. **`Notebook_4_Ablation_and_Arch.ipynb`**: Generates plots, tracks detailed training logs, and supports ablation analyses across experimental runs.

---

## 🧬 Ablation Studies

To reproduce the automated ablation studies described in the report, run:

```bash
python run_ablation_v5.py
```

This script sequentially trains multiple model configurations, including the baseline model, the full model, the model with pseudo-labeling, and the model with distillation. It then extracts evaluation metrics such as `mAP` and `mAP_50` and generates detailed performance comparison tables across incremental tasks.

---

## 📄 License

This project is released under the terms specified in the accompanying [LICENSE](LICENSE) file.

---

## 🎓 Citation and Acknowledgements

YOLOW-RSIOD integrates architectural components from [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and [MMDetection](https://github.com/open-mmlab/mmdetection). We sincerely acknowledge the contributions of the open-source community and the authors of these projects.
