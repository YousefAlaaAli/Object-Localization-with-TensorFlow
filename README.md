# 🧠 Emoji Detection & Classification with CNN

Detect and classify emojis in images using a Convolutional Neural Network (CNN) model built with TensorFlow/Keras.  
This project predicts both the **class** of an emoji and its **location** using a dual-output neural network — one for classification and one for bounding box regression.

---

## 📌 Features

- 🔍 **Emoji Detection** — Locate emoji within the image by predicting bounding box coordinates.  
- 🎭 **Emoji Classification** — Classify the emoji (e.g., happy, sad, skeptical, etc.).  
- 📦 **Custom `IoU` Metric** — Measures the overlap between predicted and ground-truth boxes.  
- 🖼️ **Visualization** — Real-time plotting of predicted vs actual results with bounding boxes.

---

## 🏗️ Model Architecture

- CNN backbone (Conv + MaxPooling layers)
- Two heads:
  - `class_out` → `softmax` activation for emoji classification
  - `box_out` → `linear` activation for bounding box coordinates  
- Compiled with:
  ```python
  model.compile(
      loss={
          'class_out': 'categorical_crossentropy',
          'box_out': 'mse'
      },
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      metrics={
          'class_out': 'accuracy',
          'box_out': IoU(name='iou')
      }
  )

---


## 🔍 Output Visualization
When testing the model, the visualization uses the following color-coding:

- **Bounding Boxes**:
  - ✅ Green box = Ground-truth location (actual position)
  - ❌ Red box = Model's predicted location

- **Class Labels**:
  - ✅ Green label = Correct classification
  - ❌ Red label = Incorrect classification

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/emoji-detector.git
cd emoji-detector
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Train the model**
```bash
jupyter notebook train.ipynb
```
4. **Test predictions**
```bash
jupyter notebook test.ipynb
```

---

## 🔧 Requirements
- Python 3.6+

- TensorFlow 2.x

- Jupyter Notebook

---

## 📊 Custom IoU Metric
Includes a custom tf.keras.metrics.Metric implementation to track bounding box overlap performance across batches.

---

## 🔮 Future Improvements
- Detect multiple emojis per image
- Add data augmentation
- Train on larger, real-world datasets
- Use YOLO or SSD for real-time detection

---

## 📌 Dependencies
TensorFlow

NumPy

Matplotlib

Pillow (PIL)
