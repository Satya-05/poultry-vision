<div align="center">

<h1>🐔 Poultry Vision – AI-Powered Poultry Disease Detection</h1>

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/TensorFlow-2.12-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Flask-2.2-red?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/Accuracy-86.5%25-success?style=for-the-badge" alt="Accuracy">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">

**Transfer Learning-Based Classification of Poultry Diseases**  
Detect Coccidiosis, Healthy, Newcastle Disease, and Salmonella from fecal images using deep learning.

[Live Demo](#live-demo) · [Dataset](#dataset) · [Results](#results) · [Run Locally](#run-locally)

</div>

## 🎯 Project Goal

Build a reliable, deployable ML web application that helps poultry farmers and veterinarians get **instant preliminary diagnosis** from fecal images — reducing time to action and improving flock health management.

## ✨ Features

- Transfer learning with VGG16 (fine-tuned)
- Web interface built with Flask
- Image upload → real-time prediction
- Confidence score + treatment recommendations
- Clean, responsive UI
- 86.5% test accuracy (balanced classes)

## 🏆 Results Summary

| Metric              | Value     |
|---------------------|-----------|
| Test Accuracy       | 86.50%    |
| Macro Avg F1-Score  | 0.87      |
| Coccidiosis F1      | 0.92      |
| Healthy F1          | 0.88      |
| Newcastle Disease F1| 0.77      |
| Salmonella F1       | 0.90      |

Confusion Matrix:

![confusion-matrix](docs/images/confusion_matrix.png)  
*(add screenshot after running evaluation)*

Classification Report:
