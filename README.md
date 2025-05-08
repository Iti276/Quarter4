# Pneumonia Detection Using Convolutional Neural Networks (CNN)

This repository contains a PyTorch-based deep learning pipeline for classifying chest X-ray images into **Pneumonia** or **Normal** categories.

---

## 📁 Dataset Structure

The model uses a chest X-ray dataset organized in the following structure:

```
chest_xray_new/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> You can download the dataset from [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🧠 Model Overview

The CNN model includes:

* Two convolutional layers with ReLU activation
* MaxPooling layers
* Dropout layer to reduce overfitting
* Two fully connected (FC) layers

The final output is a binary classification: `NORMAL` or `PNEUMONIA`.

---

## 🧪 Evaluation Metrics

After training, the model is evaluated using:

* Classification Report (precision, recall, F1-score)
* Confusion Matrix
* Visual analysis of predictions

### Example Output:

```
              precision    recall  f1-score   support

      Normal       0.91      0.56      0.69       234
   Pneumonia       0.78      0.97      0.87       390

    accuracy                           0.81       624
   macro avg       0.85      0.76      0.78       624
weighted avg       0.83      0.81      0.80       624
```

---

## 🖼️ Visualizations

The script includes functions to:

* Display correctly and incorrectly predicted samples:

  * True Positives: Predicted = 1, Actual = 1
  * True Negatives: Predicted = 0, Actual = 0
  * False Positives: Predicted = 1, Actual = 0
  * False Negatives: Predicted = 0, Actual = 1

* Show Confusion Matrix heatmap

These plots help interpret the model’s performance visually.

---

## 💻 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/pneumonia-cnn.git
cd pneumonia-cnn
```

2. **Install the dependencies:**

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

3. **Organize your dataset:** Place the chest\_xray\_new folder with the expected subfolders in the project directory.

4. **Run the training script:**

```bash
python train.py
```

---

## ⚙️ Features

* Data augmentation on the training set (random flip and rotation)
* Early stopping to avoid overfitting
* Classification evaluation using `sklearn`
* Confusion matrix visualization using `matplotlib`

---

## 🛠️ Requirements

* Python 3.7+
* PyTorch
* torchvision
* scikit-learn
* numpy
* matplotlib

---

## 👤 Author

**Iti Rohilla**
Email: [rohilla.i@northeastern.edu](mailto:rohilla.i@northeastern.ed)\\
