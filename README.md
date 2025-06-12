---
title: Plant-Disease-Detection-EfficientNetB1
emoji: 🌿
colorFrom: green
colorTo: purple
sdk: gradio
app_file: app.py
pinned: true
license: mit
tags:
  - plant-disease
  - image-classification
  - efficientnet
  - tensorflow
  - gradio
  - computer-vision
  - huggingface-spaces
---

[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?logo=huggingface&style=flat-square)](https://huggingface.co/spaces/McKlay/Plant-Disease-Detection-EfficientNetB1)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange?logo=gradio&style=flat-square)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GitHub last commit](https://img.shields.io/github/last-commit/McKlay/plant-disease-type-EfficientNetB1)
![GitHub stars](https://img.shields.io/github/stars/McKlay/plant-disease-type-EfficientNetB1?style=social)
![GitHub forks](https://img.shields.io/github/forks/McKlay/plant-disease-type-EfficientNetB1?style=social)
![MIT License](https://img.shields.io/github/license/McKlay/plant-disease-type-EfficientNetB1)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=McKlay.plant-disease-type-EfficientNetB1)

# 🌿 Plant Disease Detection with EfficientNetB1

**Plant-Disease-Detection-EfficientNetB1** is a deep learning-powered app that identifies 15 types of plant diseases from leaf images using a fine-tuned **EfficientNetB1** model trained on the PlantVillage dataset.

> 📸 Upload an image, paste from clipboard, or use **webcam input**  
> Get instant predictions on plant disease type — with confidence score and class label.

---

## 🌐 Demo

Deployed on Hugging Face Spaces:  
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/McKlay/Plant-Disease-Detection-EfficientNetB1)

---

## 🧠 Model Details

- **Model:** EfficientNetB1 (Keras, Sequential API)
- **Classes:** 15 plant disease types
- **Input Size:** 240×240  
- **Preprocessing:** Normalization (0–1), real-time data augmentation
- **Training:**
  - Phase 1: Freeze base, train classification head
  - Phase 2: Unfreeze all, fine-tune at low learning rate (`1e-5`)
- **Validation Accuracy:** ~97%

---

## 📓 Training Notebook (Kaggle)

The model was trained using TensorFlow and Keras on Kaggle.  
[🔗 fine-tuning-efficientnetb1-plantdiseasedetection](https://www.kaggle.com/code/claymarksarte/fine-tuning-efficientnetb1-plantdiseasedetection/notebook)  
Includes:
- Stratified 80/20 data split using `ImageDataGenerator`
- Real-time augmentations
- Early stopping and checkpointing (best weights only)
- Final `.h5` weights file exported

---

## Features

- Classifies **15 plant diseases**
- Supports **upload**, **webcam**, and **clipboard paste**
- Outputs **predicted class name** and **confidence score**
- Lightweight & fast inference — powered by Gradio

---

## ⚠️ Grad-CAM Note

Grad-CAM is **disabled** in this version due to using `Sequential()` with `load_weights()`  
To enable Grad-CAM, re-train with `Functional API` and access the last convolutional layer output.

---

## 📁 Folder Structure

```bash
12_PlantDiseaseDetection-HF/
├── app.py                   # Gradio interface
├── inference_utils.py       # Model loading + prediction
├── model/
│   ├── efficientnetb1_plant_final.weights.h5
│   └── class_names.json
├── assets/                  # Demo/test images
├── requirements.txt
└── README.md
````

---

## Example Output

| Input Image                                                                                                  | Prediction                       |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| ![sample](https://huggingface.co/datasets/McKlay/documentation-images/resolve/main/plant-disease-detect-EfficientNetB1/Tomato_Early_blight.jpg) | 🍅 Tomato - Early Blight (99%) |

---

## Installation

To run locally:

```bash
git clone https://github.com/McKlay/plant-disease-type-EfficientNetB1
cd plant-disease-type-EfficientNetB1
pip install -r requirements.txt
python app.py
```

---

## Requirements

```txt
tensorflow
gradio
opencv-python
numpy
Pillow
```

---

## 👨‍💻 Author

Developed by [Clay Mark Sarte](https://github.com/McKlay)  
Built with TensorFlow + Gradio  
Deployed to Hugging Face Spaces

---

## ⚠️ Disclaimer

> **This tool is for educational use only.**
> Not intended for use in real-world agricultural decisions or medical treatment.

---