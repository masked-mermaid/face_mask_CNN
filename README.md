#  Facemask Detector CNN

A simple Convolutional Neural Network (CNN) for detecting face mask usage from images and live video.

---

## 🚀 Overview

This project implements a CNN-based image classifier that categorizes faces into **three classes**:

-  **Mask**
- **Partial Mask**
-  **No Mask**

It also includes a **real-time webcam inference script** using OpenCV.

---

## Features

- Image-based training using TensorFlow / Keras
- Real-time face mask detection via webcam
- Easy-to-reproduce Conda environment
- Modular and beginner-friendly structure

---

## Tech Stack & Requirements

The project is developed using a **Conda environment** with the following dependencies:

| Library | Version | Purpose |
|------|------|------|
| **TensorFlow** | `2.15.1` | Deep learning & CNN model |
| **NumPy** | `1.26.4` | Numerical computations |
| **Matplotlib** | `3.10.8` | Training curves & plots |
| **Scikit-learn** | `1.7.2` | Evaluation metrics |
| **OpenCV** | `4.13.0` | Image & video processing |
| **Seaborn** | latest | Visualization |

---

## 📦 Dataset

Download the dataset from Kaggle:

👉 https://www.kaggle.com/datasets/jamesnogra/face-mask-usage

After downloading, extract it into the project root as:

```text
dataset/
├── with_mask/
├── partial_mask/
└── without_mask/
```

##  📂 Project Structure
```text
├── dataset/                    # Image dataset
├── train.ipynb                 # Model training notebook
├── live_video_labeling.py      # Real-time webcam inference
├── mask_model.keras            # Trained CNN model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
## ⚙️ Installation (User Machine Setup)
1️⃣ Clone the Repository
```bash
git clone https://github.com/masked-mermaid/face_mask_CNN.git
cd face_mask_CNN
```

## 2️⃣ Create Conda Environment
```bash
conda create -n facemask python=3.10 -y
conda activate facemask
```

## 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```


# 🧪 Training the Model

##Open the Jupyter notebook:

```bash 
jupyter notebook
```

## Then run:

```bash
train.ipynb
```

This will:

Load the dataset

Train the CNN

Save the trained model as mask_model.h5

## 🎥 Run Real-Time Face Mask Detection

Make sure the trained model exists, then run:

```bash
python live_video_labeling.py
```

📸 Press q to exit the webcam window.
