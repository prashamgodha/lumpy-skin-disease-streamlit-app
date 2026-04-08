# 🐄 Lumpy Skin Disease Detection using Deep Learning

A deep learning-based livestock disease detection system leveraging a **pre-trained CNN models**, transfer learning, and image classification for accurate detection of Lumpy Skin Disease in cattle.

🚀 **Live App:** https://lumpy-skin-disease-app-app-eux4j4byzabeo9d4rrc9sp.streamlit.app

📂 **GitHub Repo:** https://github.com/prashamgodha/lumpy-skin-disease-streamlit-app.git  

---

## 📌 Overview

This project presents a **complete end-to-end deep learning pipeline** for automated Lumpy Skin Disease detection in livestock.

The system:
- Classifies images into **Normal Skin vs Lumpy Skin Disease**  
- Uses **transfer learning with pretrained CNN architectures**  
- Compares multiple models for performance evaluation  
- Provides predictions via a **Streamlit web interface**  

It offers a **scalable, cost-effective, and AI-powered solution** for early livestock disease detection.

---

## 🎯 Key Features

- 🖼️ Image-based disease classification  
- 🧠 Deep learning using pretrained CNN models  
- ⚖️ Comparative analysis of multiple architectures  
- 📊 High accuracy with EfficientNetB0  
- 🌐 Interactive Streamlit web interface  
- ⚡ Fast and efficient predictions  
- 🚜 Suitable for real-world field deployment  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - TensorFlow / Keras  
  - OpenCV  
  - NumPy, Pandas  
  - Scikit-learn  
  - Matplotlib / Seaborn  
  - Streamlit  
- **Model:** Pre-trained CNNs (MobileNetV2, ResNet50, EfficientNetB0)

---

## ⚙️ System Architecture

1. Upload livestock skin image  
2. Resize image to 224×224  
3. Apply preprocessing  
4. Extract features using CNN backbone  
5. Pass features through classification head  
6. Generate probability score  
7. Output prediction (Normal / Lumpy Skin)  

---

## 🧠 Model Selection

| Model     | Accuracy | Scalability | Robustness | Final Choice |
|----------|---------|------------|------------|-------------|
| MobileNetV2  | ~92.3% | High       | High       | Good        |
| ResNet50     | ~88.5% | Moderate   | Moderate   | Not Used    |
| EfficientNetB0 | ~94.6% | High     | High       | ⭐ Selected  |

---

## 📊 Methodology

### 🔹 Data Preprocessing
- Image resizing to 224×224  
- Normalization based on model requirements  
- Data augmentation (rotation, zoom, flip)  

### 🔹 Model Training
- Transfer learning using ImageNet weights  
- Two-phase training:
  - Frozen backbone training  
  - Fine-tuning top layers  

### 🔹 Classification
- Binary classification (Normal vs Lumpy Skin)  
- Sigmoid activation for probability output  

### 🔹 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

---

## 🖥️ Application Features

### 📌 Modules

- Upload Image → Input livestock image  
- Select Model → Choose CNN architecture  
- Predict → Run classification  
- Output → Show result with confidence  

---

## 📊 Results & Performance

- ✅ EfficientNetB0 achieved highest accuracy (~94.6%)  
- ⚡ Fast inference suitable for real-time use  
- 🎯 High precision and recall for disease detection  
- 📊 Strong generalization on validation data  
- 📉 Slight performance drop on low-quality images  

---

## 🏆 Key Achievements

- Built **complete deep learning pipeline for livestock disease detection**  
- Implemented **comparative analysis of CNN architectures**  
- Achieved **high accuracy using EfficientNetB0**  
- Designed **user-friendly Streamlit interface**  
- Developed **real-world deployable AI solution**  

---

## 👨‍💻 Team Members

- Krish Naik  
- Nrependre Shivhare  
- Prasham Godha  

---

## 🙏 Mentors

- Dr. K. K. Sharma  
- Dr. Lalit Purohit  
- Dr. Upendra Singh  
- Mr. Akshay Gupta  

---

## 🔮 Future Work

- Multi-class livestock disease detection  
- Mobile app deployment (TensorFlow Lite)  
- Integration with veterinary systems  
- Explainable AI (Grad-CAM visualization)  
- Larger and diverse dataset training  

---

## 📚 References

- MobileNetV2 (Google Research)  
- ResNet (Microsoft Research)  
- EfficientNet (Google AI)  
- TensorFlow Documentation  
- Research papers on Deep Learning in Agriculture  

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
