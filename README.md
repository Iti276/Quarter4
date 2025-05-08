Pneumonia Detection Using Convolutional Neural Networks (CNN)
This repository contains a deep learning-based pipeline built with PyTorch to detect pneumonia from chest X-ray images. The project includes image preprocessing, model training with early stopping, evaluation, and visualization of results using a custom Convolutional Neural Network.

🧠 Model Summary
The model is a simple CNN architecture trained on a chest X-ray dataset (binary classification: Normal vs. Pneumonia). The pipeline uses data augmentation for training and implements early stopping to avoid overfitting.

📁 Dataset
The dataset is structured using the ImageFolder format with the following directory layout:

css
Copy
Edit
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
📌 The dataset can be downloaded from the Kaggle Chest X-Ray Images (Pneumonia) dataset.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/pneumonia-cnn.git
cd pneumonia-cnn
Install required packages:

bash
Copy
Edit
pip install torch torchvision scikit-learn matplotlib
Place the dataset under the correct directory structure (see above).

Run the main training script:

bash
Copy
Edit
python train.py
🔍 Key Features
Data Augmentation: Random rotations and horizontal flips for generalization.

CNN Architecture:

2 convolutional layers

Max pooling and dropout

Fully connected layers for classification

Early Stopping: Based on validation loss to prevent overfitting.

Evaluation:

Classification report (Precision, Recall, F1-score)

Confusion matrix

Visualizations of correctly and incorrectly classified images

📊 Sample Results
Classification Report:

mathematica
Copy
Edit
              precision    recall  f1-score   support
      Normal       0.91      0.56      0.69       234
   Pneumonia       0.78      0.97      0.87       390
   Accuracy                            0.81       624
Confusion Matrix:


Sample Visualizations:

Correct and incorrect classifications shown using matplotlib for model interpretation.

📌 Requirements
Python 3.7+

PyTorch

torchvision

scikit-learn

matplotlib

numpy