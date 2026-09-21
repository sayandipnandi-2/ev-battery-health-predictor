# ⚡ EV Battery Health Predictor

A machine learning project that classifies the health of an electric-vehicle Li-ion battery as **Healthy**, **Degrading** or **Critical** using the NASA Li-ion battery dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange) ![Status](https://img.shields.io/badge/status-complete-brightgreen)

## 📌 Overview
Battery degradation affects the safety, range and resale value of EVs. This project derives **State of Health (SoH)** from discharge capacity and trains a **Random Forest classifier** on features engineered from discharge curves, then tests it on batteries the model has never seen.

## ✨ Highlights
- Uses NASA Li-ion battery data: **4 batteries, 636 discharge cycles**
- SoH derived from discharge capacity and mapped to 3 health classes
- Features engineered from discharge curves: time to voltage thresholds and temperature
- **Leakage prevention:** inputs such as cycle number removed so predictions rely on real battery behaviour
- **Leave-one-battery-out validation** on unseen batteries
- Showed that a **random split overestimates accuracy** compared with battery-wise testing
- Final model saved with `joblib`

## 🧰 Tech Stack
Python · Pandas · NumPy · Matplotlib · Scikit-learn · Joblib

## 📂 Project Structure
```
ev-battery-health-predictor/
├── dataset/
│   └── cleaned_dataset/        # metadata.csv and cycle data files
├── outputs/                    # charts and saved model
├── ev_battery_healt_predictor.py   # main script
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔬 Methodology
1. **Data preparation:** load `metadata.csv`, summarise each discharge cycle and compute SoH from capacity.
2. **Labelling:** map SoH to Healthy / Degrading / Critical.
3. **Feature engineering:** time to voltage thresholds and temperature features.
4. **Leakage control:** drop cycle number and similar age-revealing inputs.
5. **Modelling:** Random Forest classifier.
6. **Validation:** leave-one-battery-out, compared with a random split.
7. **Evaluation:** accuracy, classification report and confusion matrix.

## 🚀 Getting Started
```bash
git clone https://github.com/sayandipnandi-2/ev-battery-health-predictor.git
cd ev-battery-health-predictor
pip install -r requirements.txt
python ev_battery_healt_predictor.py
```

## 🔭 Future Work
- Test on more batteries and chemistries
- Predict Remaining Useful Life (RUL)
- Build a simple monitoring dashboard

## 📚 Data Source
NASA Prognostics Center of Excellence, *Li-ion Battery Data Set* (B. Saha and K. Goebel, 2007).

## 👤 Author
**Sayandip Nandi** — B.Tech ECE, Institute of Engineering and Management, Kolkata
[LinkedIn](https://www.linkedin.com/in/sayandip2004/) · [GitHub](https://github.com/sayandipnandi-2) · sayandipnandi0@gmail.com
