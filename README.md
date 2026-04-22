# Gesture Sense - Automated Deep Learning Based Gesture Control

## 📌 Project Overview

**Gesture Sense - Automated Deep Learning Based Gesture Control** is a real-time computer vision system that enables users to control their laptop using hand gestures without physical interaction.

This project uses a **deep learning-based approach** to recognize gestures from video sequences and map them to system-level actions such as volume control, navigation, and mouse clicks.

---

## 🚀 Key Features

* 🎥 Real-time gesture recognition using webcam
* 🧠 Hybrid Deep Learning Model (**3D CNN + ConvLSTM**)
* ⚡ Ensemble learning for improved prediction accuracy
* 🖥️ Laptop control using keyboard and mouse automation
* 🔍 Confidence-based filtering to reduce false predictions
* 🔄 Model training + export + reuse workflow

---

## 📂 Project Structure

```
.
├── ModelTraining.ipynb        # Model training, evaluation, and export
├── realtime.py               # Real-time gesture detection & inference
├── switch1.py                # Gesture → system control mapping
├── class_jester_6_300.csv    # Gesture labels
├── README.md                 # Project documentation
├── LICENSE
└── .gitignore
```

---

## 📊 Dataset & Input Format

* Gesture dataset consists of **video frame sequences**
* Each sample:

  * **16 frames per gesture**
  * Resized to **64 × 64**
  * RGB format

### 🧾 Classes

* Backward
* Forward
* Stop
* Swiping Left
* Swiping Right
* No Gesture

### 🔹 Input Shape

```
(16, 64, 64, 3)
```

---

## 🧠 Model Architecture

This project uses a **hybrid deep learning architecture**:

### 🔹 1. 3D CNN (Feature Extraction)

* Captures **spatial + short-term temporal features**
* Learns:

  * Motion patterns
  * Hand structure
  * Gesture dynamics

### 🔹 2. ConvLSTM (Temporal Modeling)

* Captures **long-term dependencies across frames**
* Maintains temporal memory of gestures

### 🔹 3. Classification Head

* Global Average Pooling
* Dense layer
* Softmax activation (multi-class output)

---

## ⚙️ Training Details

| Parameter      | Value                        |
| -------------- | ---------------------------- |
| Loss Function  | Categorical Crossentropy     |
| Optimizer      | SGD (lr=0.005, momentum=0.9) |
| Batch Size     | 32                           |
| Epochs         | 300                          |
| Regularization | Dropout (0.5), L2            |

---

## 💾 Model Training & Export

The entire training pipeline is implemented in:

```
ModelTraining.ipynb
```

### 🔹 Workflow:

1. Load and preprocess dataset
2. Train model (3D CNN + ConvLSTM)
3. Evaluate performance (accuracy, confusion matrix)
4. Save trained model

### 🔹 Model Export:

```python
model.save("saved_model/model_name")
```

### 🔹 Model Import (for inference):

```python
from keras.models import load_model
model = load_model("saved_model/model_name")
```

👉 This allows the trained model to be reused without retraining.

---

## 🔄 Ensemble Learning

Two trained models are combined to improve performance:

```
Final Prediction = Model1 + Model2
```

* Reduces variance
* Improves robustness
* Handles edge cases better

---

## 🎥 Real-Time Inference Pipeline

1. Capture webcam frames using OpenCV
2. Store frames in buffer (16 frames)
3. Preprocess frames (resize + normalize)
4. Load trained models
5. Predict gesture using both models
6. Combine predictions (ensemble)
7. Apply confidence threshold filtering
8. Display predicted gesture
9. Trigger system action

---

## 🖱️ Gesture to System Mapping

| Gesture       | Action         |
| ------------- | -------------- |
| Forward       | Volume Up      |
| Backward      | Volume Down    |
| Stop          | Mouse Click    |
| Swiping Left  | Previous Track |
| Swiping Right | Next Track     |

---

## 🧰 Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Pynput

---

## 📈 Evaluation

* Accuracy: ~75%
* Evaluated using:

  * Confusion Matrix
  * Training & Validation Curves

### 🔹 Observations:

* Similar gestures can cause misclassification
* Performance depends on lighting and background

---

## ⚠️ Limitations

* Fixed frame window introduces latency
* Sensitive to lighting conditions
* Reduced performance for very fast gestures

---

## 🔮 Future Improvements

* Use Transformer-based video models
* Optimize inference using TensorRT / ONNX
* Improve dataset diversity
* Reduce latency with dynamic frame windows

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install opencv-python tensorflow keras numpy pynput matplotlib
```

### 2. Run Real-Time Gesture Control

```bash
python realtime.py
```

---

## 🧠 Key Highlights

* Spatio-temporal modeling using **3D CNN + ConvLSTM**
* Real-time inference system
* Ensemble learning for performance improvement
* End-to-end pipeline: **training → export → deployment**

---

## 👨‍💻 Author

**Nikith Gokul**

---

