# Supervised Contrastive Learning & Episodic Test-Time Adaptation for Few-Shot Fish Classification

This repository contains the implementation of a two-stage framework combining **Supervised Contrastive Pre-training (SupCon)** with **Episodic Test-Time Adaptation (TTA)** for fine-grained fish species classification across underwater datasets.

---

## 📊 Dataset Availability

Please download the datasets from their official sources/mirrors below before running the training pipeline:

| Dataset | Source / Download Link | Description |
| :--- | :--- | :--- |
| **Croatian** | [Kaggle Dataset](https://www.kaggle.com/datasets/ashfaqsyed/croatian-fish-dataset) | Underwater fish species collected in the Adriatic Sea. |
| **Fish4Knowledge (F4K)** | [University of Edinburgh Source](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/) | Verified underwater fish images from live video feeds in Taiwan. |
| **Aqua20** | [AQUA20 GitHub Repository](https://github.com/taufiktrf/AQUA20) | Aquatic species benchmark designed for complex underwater environments. |

---

## 📂 Data Directory Setup

After downloading and extracting the datasets, place them into the relative `./data` directory at the root of this project. 

Organize your folder structure as follows so the data loader can parse class folders automatically:

```text
fish-fewshot-supcon/
├── data/
│   ├── Croatian/
│   │   ├── class_1/
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   └── class_2/
│   ├── F4K/
│   │   ├── class_1/
│   │   └── class_2/
│   └── Aqua20/
│       ├── class_1/
│       └── class_2/
├── checkpoints/
├── config.py
├── datasets.py
├── models.py
├── trainer.py
├── utils.py
├── main.py
└── requirements.txt
