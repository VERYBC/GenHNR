# 🚀 GenHNR: Significantly Boost the Retrieval Accuracy of Your UVGL Method

This repository provides the dataset and code for our paper **GenHNR: A Generalizable Hard Negative Refinement Framework for UAV-View Geo-Localization**. Thank you for your interest in our work!

---

## 🗺️ 1. XIHUA-Visloc Dataset

We introduce **XIHUA-Visloc**, a real-world continuous-scene UAV localization dataset designed following the principles of XIAN-Visloc. Unlike XIAN-Visloc, which primarily covers urban environments, XIHUA-Visloc focuses on **town environments** in [Xihua County, Zhoukou, China](https://www.google.com/maps/search/?api=1&query=Xihua%20County%2C%20Zhoukou%2C%20Henan%2C%20China). These environments are characterized by dense, low-rise buildings and sparsely distributed farmland.

The dataset consists of eight flight sequences captured using a DJI Air 3S UAV, with a total flight distance of **42.32 km** and flight altitudes ranging from **300 to 350 m**. It contains **3,555 UAV images** at a resolution of $3840 \times 2160$ pixels. Four regional satellite maps obtained from Google Earth cover a total area of **55.09 km²**.

The first six flight sequences are used for training, while the remaining two are reserved for testing.

<p align="center">
  <img src="docs/XIHUA-Visloc.png" alt="Overview of the XIHUA-Visloc dataset" width="50%">
</p>

---

## 🛠️ 2. Data Preparation

### 📥 2.1. Download the Dataset

Download [**XIHUA-Visloc from Hugging Face**](https://huggingface.co/datasets/VERYBC/XIHUA-Visloc) and place the dataset files in the following directory:

```text
data/Xihua_visloc/
```

### 🛰️ 2.2. Build the Satellite Reference Galleries

Run the following scripts to construct the satellite reference galleries for training and testing:

```bash
python scripts/XIHUA_Visloc_train.py
python scripts/XIHUA_Visloc_test.py
```

### 🖼️ 2.3. Resize the UAV Images

Run the following script to resize the UAV images:

```bash
python scripts/UAV_image_rescaling.py
```

The processed UAV images will be saved in the `drone_392/` directory.

---

## 🧪 3. Evaluation on XIHUA-Visloc

### 📦 3.1. Download Model Weights

Download the pretrained model weights from [**Google Drive**](https://drive.google.com/drive/folders/1E2IXkT4bqmuNL_bT2Fo20K0c39E1pOCI?usp=drive_link) and place the weight files in the `checkpoints/` directory.

### 🔍 3.2. Stage 1: Cross-View Retrieval

Run the following script to perform initial cross-view retrieval:

```bash
python Stage_1_retrieval_xihua.py
```

The initial retrieval results will be saved in the `retrieval_results/` directory.

### 🎯 3.3. Stage 2: Fine-Grained Re-ranking

Run the following script to refine the initial retrieval results:

```bash
python Stage_2_reranking_xihua.py
```

> **📌 Note**
>
> In the released implementation, clustering and geographic aggregation are not enabled in Stage 1. This setting is controlled by the `if_SC_GA` option in `Stage_1_retrieval_xihua.py`. Consequently, the re-ranking results obtained with this implementation may be better than those reported in the paper.

---

## 🙏 4. Acknowledgements

This repository builds upon the following open-source projects:

- 🔗 [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
- 🔗 [IHN](https://github.com/imdumpl78/IHN)
- 🔗 [HCNet](https://github.com/xlwangDev/HC-Net)

We sincerely thank the authors for making their code publicly available.
