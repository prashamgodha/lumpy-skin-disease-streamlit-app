# 🐄 LSD Detect — Lumpy Skin Disease Detection Dashboard

A production-grade Streamlit dashboard for multi-model Lumpy Skin Disease detection 
using MobileNetV2, ResNet50, and EfficientNetB0 with transfer learning + fine-tuning.

---

## 🚀 Quick Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
gh repo create lumpy-skin-detect --public --source=. --push
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** ✅

---

## 🖥️ Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 To Use Real Models (Production Mode)

Place your trained model files alongside `app.py`:
```
lumpy_skin_dashboard/
├── app.py
├── requirements.txt
├── fmobilenet_lumpy.h5
├── fresnet_lumpy.h5         ← rename from fresnet_lumpy.pkl
├── fefficientnet_lumpy.h5   ← rename from fefficientnet_lumpy.pkl
└── class_names.pkl
```

Then in `app.py`, replace the simulated prediction block with:
```python
@st.cache_resource
def load_all_models():
    from tensorflow.keras.models import load_model
    mob = load_model("fmobilenet_lumpy.h5")
    res = load_model("fresnet_lumpy.h5")
    eff = load_model("fefficientnet_lumpy.h5")
    return mob, res, eff

model_mob, model_res, model_eff = load_all_models()

def preprocess(image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = image.resize((224,224))
    img = np.array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, 0)

img = preprocess(uploaded_image)
pred_mob = float(model_mob.predict(img)[0][0])
pred_res = float(model_res.predict(img)[0][0])
pred_eff = float(model_eff.predict(img)[0][0])
```

> ⚠️ Note: Streamlit Cloud free tier has 1 GB RAM. TF models (~100MB+) 
> may hit limits. Consider Hugging Face Spaces (2GB) or Railway for larger models.

---

## 🌐 Alternative Deployment Options

| Platform | Free Tier | Notes |
|---|---|---|
| **Streamlit Cloud** | ✅ Yes | 1 GB RAM, easiest |
| **Hugging Face Spaces** | ✅ Yes | 2 GB RAM, great for ML |
| **Railway** | Partial | Better for heavy models |
| **Render** | ✅ Yes | 512MB RAM free |

---

## 📊 Dashboard Features

- **🔍 Predict Tab** — Upload image, get predictions from all 3 models with confidence bars
- **📊 Model Comparison** — Accuracy, Precision, Recall, F1, AUC-ROC bar charts + ROC curves
- **📈 Training Curves** — Epoch-by-epoch accuracy and loss plots per model
- **🧩 Confusion Matrix** — Interactive per-model confusion matrices with classification report
- **🏗️ Architecture** — Visual layer-by-layer architecture diagram + training pipeline

---

## 🎛️ Sidebar Controls

- Upload cattle skin images (JPG/PNG)
- Adjust decision threshold (default 0.5, best threshold for ResNet50: 0.36)
- Toggle which models participate in ensemble prediction
