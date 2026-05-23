# 📨 Spam Email & SMS Detector (End-to-End AI Pipeline)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![Framework](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io)
[![UI Interface](https://img.shields.io/badge/UI--Interface-Gradio-orange)](https://github.com/gradio-app/gradio)

An end-to-end deep learning pipeline that classifies text messages (SMS and short emails) as **Spam** or **Ham** (legitimate). Uses a modern, production-ready NLP approach by embedding all preprocessing layers directly inside the neural network graph, backed by a Bidirectional LSTM architecture.

---

## 🚀 Key Features & Performance

- **High Accuracy:** Achieves ~98% test accuracy on unseen data.
- **Self-Contained Preprocessing:** Uses Keras `TextVectorization` inside the model graph — no external tokenizer files needed at inference time.
- **Context-Aware Bidirectional LSTM:** Evaluates context from both directions simultaneously for stronger detection of complex spam phrasing.
- **Auto-Tuned Decision Threshold:** A precision-recall sweep finds the optimal classification threshold instead of defaulting to 0.5. This minimises false positives (legitimate messages incorrectly marked as spam).
- **Smart LR Scheduling:** `ReduceLROnPlateau` automatically halves the learning rate when training plateaus, squeezing out the last fraction of accuracy without manual tuning.
- **Persistent Checkpoints:** Model checkpoints are saved to Google Drive so the best weights survive Colab runtime resets.
- **Interactive Web App:** Gradio UI with `share=True` generates a real public URL you can open on any device.

---

## 🛠️ Tech Stack & Architecture

### Frameworks & Libraries

- **Core:** Python 3.8+
- **Deep Learning:** TensorFlow & Keras
- **Data Processing:** NumPy, Pandas, Scikit-Learn
- **Visualisation & UI:** Seaborn, Matplotlib, Gradio

### NLP Pipeline & Model Structure

Raw text flows through the graph like this:

1. **Text Input Layer** — accepts raw, variable-length strings (`tf.string`, `shape=()`)
2. **Keras TextVectorization** — tokenises, lowercases, and pads to a fixed sequence length
3. **Masked Word Embeddings** — maps token integers to dense 128-d vectors; `mask_zero=True` tells the LSTM to ignore padding
4. **Bidirectional LSTM (64 units)** — extracts sequential context forwards and backwards simultaneously
5. **Global Max Pooling (1D)** — collapses the sequence to a single vector, keeping the strongest spam signal
6. **Dropout + Dense** — regularises with 0.4 / 0.2 dropout rates before the classification layer
7. **Output Node** — sigmoid activation returns a single probability in [0, 1]

### Dataset

- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size:** 5,572 messages — 4,827 ham (87%) and 745 spam (13%)
- Computed class weights handle the imbalance during training.

---

## 📈 Results & Evaluation

| Metric | Score |
| :--- | :--- |
| Test accuracy | ~98% |
| Precision (spam) | ~97% |
| Recall (spam) | ~94% |
| Decision threshold | Auto-tuned via PR sweep |

The notebook includes:

- Training history plots (accuracy, loss, learning rate per epoch)
- Confusion matrix in both raw counts and row-normalised percentages
- Precision-recall curve with the optimal threshold marked

---

## 💻 How to Run

### Option 1: Google Colab (recommended)

1. Upload `spam_detector_v2.ipynb` to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Run all** (or `Ctrl+F9`).
3. When prompted, allow Colab to mount your Google Drive (Step 1b). This keeps your checkpoints safe.
4. Scroll to the last cell to interact with the live Gradio app via the public share link.

### Option 2: Local setup

```bash
# Clone the repository
git clone https://github.com/yourusername/spam-email-detector.git
cd spam-email-detector

# Install dependencies
pip install tensorflow pandas numpy scikit-learn seaborn gradio jupyter

# Launch
jupyter notebook spam_detector_v2.ipynb
```

When running locally, set `USE_DRIVE = False` in Step 1b — checkpoints will save to the current directory instead.
