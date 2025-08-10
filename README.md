# Fake Speech Detection

A deep-learning system that detects whether a given audio clip is **real (human)** or **fake (synthetic / voice-converted)** using a hybrid Conv1D + LSTM architecture. The implementation, dataset details, training notes, and inference examples are documented in the included notebook and project document.

---

## Highlights
- Hybrid Conv1D → LSTM model to capture local spectral features and long-range temporal patterns.  
- Uses 26 engineered acoustic features (e.g., MFCCs, chroma, spectral centroid, zero-crossing rate).  
- Trained on a balanced real vs. synthetic dataset from sources such as LibriSpeech / Common Voice (real) and WaveNet / Tacotron2 / GAN-based generators (fake).  
- Real-time prediction pipeline: extracts features from a `.wav`, standardizes them, chunks into 5-second windows, and outputs a probability score (real vs fake).

See the notebook for training curves, confusion matrix, and detailed evaluation.

---

## Results (reported)
- **Accuracy**: ~98.87%  
- **Precision**: ~98.86%  
- **Recall**: ~98.86%  
- **F1-score**: ~98.86%  
- **External test set**: ~92.4% accuracy  
- **Inference time**: ~250 ms per 5 s clip

---

## Notebook Walkthrough

Follow the sections in this order to reproduce the project:

1. **Load Feature Dataset**  
   Load the pre-extracted 26-feature dataset (e.g., MFCCs, chroma, spectral centroid, zero-crossing rate).

2. **Preprocessing**  
   - Apply `StandardScaler` to normalize features consistently.  
   - Chunk audio into **5-second** windows to match training input format.

3. **Build Model**  
   Construct the hybrid deep-learning model:  
   - Conv1D → Conv1D → LSTM → Dense  
   - Use Batch Normalization and Max Pooling after Conv layers, followed by LSTM layers and Dense layers ending with a sigmoid activation.

4. **Training**  
   - Use **Adam** optimizer (learning rate **~0.001**), **binary cross-entropy** loss.  
   - Train with batch size **32**, up to **50 epochs** (with early stopping, typically halts ~20 epochs).  
   - Use **stratified split**: 70 % training, 15 % validation, and 15 % test.

5. **Evaluation & Export Model**  
   - Evaluate on test set: compute accuracy, precision, recall, F1-score.  
   - Export model (e.g., `model.h5`) and the fitted scaler for later inference.

---

## Inference Pipeline

To make real-time predictions on new audio:

1. **Load Saved Artifacts**  
   - Load the exported **model** and **StandardScaler** from training.

2. **Feature Extraction**  
   - Extract the same 26 acoustic features from new `.wav` input.

3. **Preprocess & Chunk**  
   - Normalize features using the loaded scaler.  
   - Chunk into **5-second** windows (same as training).

4. **Predict & Output**  
   - Run the model to get prediction for each chunk.  
   - Output label(s) (e.g., real vs. fake) along with confidence score(s).

---

## Model & Training Details

- **Architecture**:  
  - Conv1D layers (e.g., 64 → 128 filters) with BatchNorm + MaxPool  
  - Two LSTM layers (e.g., 128 → 64 units)  
  - Dense layers with L2 regularization + Dropout  
  - Final sigmoid activation for binary classification.

- **Optimizer & Loss**:  
  - **Adam** optimizer, learning rate ~0.001  
  - **Binary cross-entropy** loss function

- **Training Setup**:  
  - Batch size: 32  
  - Early stopping with patience = 5  
  - Max epochs: 50 (typically stops around 20)

- **Data Split**:  
  - Training: 70 %  
  - Validation: 15 %  
  - Test: 15 %  
  - Use stratified sampling to preserve class distribution.

---

## Tips & Gotchas

- **Scaler Consistency**: Always use the *exact* `StandardScaler` fitted during training for inference. Failure to do so invalidates predictions.

- **Chunk Length Must Match**: Consistently use **5-second** windows in both training and inference. Deviating will break input alignment and degrade performance.

- **Generalization Check**: Test your model on external datasets (e.g., WaveFake, LJ Speech) to assess robustness beyond the original data distribution.

---


