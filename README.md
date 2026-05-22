# 📨 Spam Email & SMS Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![Framework](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io)

An end-to-end deep learning pipeline that classifies text messages (SMS and short emails) as **Spam** or **Ham** (legitimate). This model leverages modern Natural Language Processing (NLP) techniques to vectorize text and capture semantic meaning, achieving high-accuracy classification.

---

## 🚀 Key Features & Performance
*   **High Accuracy:** Achieves **~98% test accuracy** on unseen data.
*   **Robust Text Preprocessing:** Implements custom tokenization, sequence padding, and word embeddings to handle messy, real-world text.
*   **Colab Ready:** Fully self-contained notebook format for zero-setup execution.

---

## 🛠️ Tech Stack & Architecture

### Frameworks & Libraries
*   **Core:** Python
*   **Deep Learning:** TensorFlow & Keras
*   **Data Processing:** NumPy, Pandas

### NLP Pipeline & Model Structure
1.  **Tokenization:** Converts raw text strings into numerical token streams based on dataset vocabulary.
2.  **Padding:** Standardizes input vector dimensions using sequence padding (`post` padding) to ensure uniform shape for network ingestion.
3.  **Embedding Layer:** Maps integer-encoded words into a dense vector space to capture semantic relationships.
4.  **Classification Layers:** Deep neural network layers optimized with the `Adam` optimizer and binary cross-entropy loss.

### Dataset
*   **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
*   **Dataset Size:** 5,572 mobile messages labeled as ham (4,827) or spam (747).

---

## 📈 Results & Evaluation

The model demonstrates strong convergence with minimal overfitting. 

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | ~99.2% |
| **Test Accuracy** | **~98.0%** |
| **Primary Focus** | Minimizing False Positives (marking legitimate messages as spam) |

---

## 💻 How to Run

You can get this model running in under two minutes using Google Colab.

### Option 1: Direct Run via Colab
1. Upload the project `.ipynb` notebook file to your [Google Drive](https://drive.google.com).
2. Open the notebook using **Google Colab**.
3. Go to **Runtime** -> **Run all** (or press `Ctrl + F9`).
4. *Note: The notebook is configured to automatically download and extract the dataset via script, so no manual file uploads are required.*

### Option 2: Local Setup
If you prefer running it locally via Jupyter Notebook or standard Python:

```bash
# Clone the repository
git clone [https://github.com/yourusername/spam-email-detector.git](https://github.com/yourusername/spam-email-detector.git)
cd spam-email-detector

# Install required dependencies
pip install tensorflow pandas numpy scikit-learn jupyter

# Launch the notebook
jupyter notebook
