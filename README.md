# 📨 Spam Email & SMS Detector (End-to-End AI Pipeline)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![Framework](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io)
[![UI Interface](https://img.shields.io/badge/UI--Interface-Gradio-orange)](https://github.com/gradio-app/gradio)

An end-to-end deep learning pipeline that classifies text messages (SMS and short emails) as **Spam** or **Ham** (legitimate). This upgraded version showcases a modern, production-ready Natural Language Processing (NLP) approach by embedding text preprocessing layers directly into the neural network graph, backed by a sequential Bidirectional LSTM architecture.

---

## 🚀 Key Features & Performance
* **High Accuracy:** Achieves **~98% test accuracy** on unseen data.
* **Self-Contained Preprocessing:** Swapped out deprecated legacy tokenizers for Keras `TextVectorization`. The model accepts raw text strings directly—eliminating external dependency files during deployment.
* **Context-Aware Bidirectional LSTM:** Uses a bidirectional network structure to evaluate context from both directions simultaneously, boosting detection of complex spam strings.
* **Interactive Web App UI:** Features a native local web deployment powered by **Gradio** allowing users to instantly copy an inference public link to test their own messages from a mobile device or external browser.
* **Colab Ready:** Fully self-contained notebook format for zero-setup execution.

---

## 🛠️ Tech Stack & Architecture

### Frameworks & Libraries
* **Core:** Python
* **Deep Learning:** TensorFlow & Keras
* **Data Processing:** NumPy, Pandas, Scikit-Learn
* **Visualization & UI:** Seaborn, Matplotlib, Gradio

### Upgraded NLP Pipeline & Model Structure
The graph processes raw, un-tokenized incoming streams structurally like this:
1.  **Text Input Layer:** Accepts raw, variable-length text sequences (`tf.string`) directly.
2.  **Keras TextVectorization Layer:** Dynamically tokenizes, lowercases, and standardizes phrases to structural array dimensions.
3.  **Masked Word Embeddings:** Maps numerical integers into dense vector spaces while utilizing `mask_zero=True` to signal the network to ignore meaningless trailing padding characters.
4.  **Bidirectional LSTM (64 Units):** Extracts historical contextual features forwards and backwards simultaneously.
5.  **Global Max Pooling (1D):** Down-samples the feature vectors to isolate the most critical semantic spam indicators.
6.  **Regularization & Classification (Dropout 0.4 / 0.2):** Minimizes network reliance on static parameters, suppressing model overfit before passing features to Dense classification nodes.

### Dataset
* **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* **Dataset Size:** 5,572 mobile messages labeled as ham (4,827) or spam (747). Includes computed balanced class weights to handle dataset skew during training.

---

## 📈 Results & Evaluation

The model demonstrates strong convergence with minimal overfitting. Because spam detection relies heavily on not blocking critical legitimate messages accidentally, we optimize for **Precision** and **Recall** alongside standard structural metrics:

| Metric | Score / Priority | Target Objective |
| :--- | :--- | :--- |
| **Training Accuracy** | ~99.2% | Model Learning Convergence |
| **Test Accuracy** | **~98.0%** | Evaluation on Unseen Data |
| **Primary Focus** | Precision & Recall | Minimizing False Positives (marking legitimate messages as spam) |

---

## 💻 How to Run

You can get this model running in under two minutes using Google Colab or your local machine.

### Option 1: Direct Run via Colab
1. Upload the project `.ipynb` notebook file to your [Google Drive](https://drive.google.com).
2. Open the notebook using **Google Colab**.
3. Go to **Runtime** -> **Run all** (or press `Ctrl + F9`).
4. Scroll down to the final cell to interact directly with the live web application!
5. *Note: The notebook is configured to automatically download and extract the dataset via script, so no manual file uploads are required.*

### Option 2: Local Setup
If you prefer running it locally via Jupyter Notebook or standard Python:

```bash
# Clone the repository
git clone [https://github.com/yourusername/spam-email-detector.git](https://github.com/yourusername/spam-email-detector.git)
cd spam-email-detector

# Install required dependencies
pip install tensorflow pandas numpy scikit-learn seaborn gradio jupyter

# Launch the notebook
jupyter notebook
