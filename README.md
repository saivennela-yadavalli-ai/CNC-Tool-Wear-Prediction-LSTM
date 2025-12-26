# CNC Tool Wear Prediction using LSTM

## 📌 Project Overview
This project implements an end-to-end **Predictive Maintenance System** to predict CNC tool wear conditions using **time-series sensor data**. A deep learning **LSTM (Long Short-Term Memory)** model is trained to analyze machining parameters and sensor signals to determine tool condition and inspection outcomes.

The solution is deployed as an interactive **Streamlit web application** on **AWS EC2**, enabling real-time prediction and visualization.

---

## 🎯 Objectives
- Predict **Tool Condition** (Good / Worn / Damaged)
- Identify whether **Machining is Finalized**
- Predict **Visual Inspection Outcome**
- Apply statistical tests for feature relevance
- Deploy the model using a scalable cloud environment (AWS)

---

## 🧠 Techniques & Concepts Used
- Time Series Analysis
- Exploratory Data Analysis (EDA)
- Outlier Handling (Clipping vs Removal)
- Statistical Tests:
  - Two-sample t-test
  - ANOVA
  - Chi-square test
- Feature Engineering
- LSTM Deep Learning Model
- Model Evaluation Metrics
- Streamlit UI Development
- AWS EC2 Deployment
- GitHub Version Control

---

## 🗂️ Project Structure
CNC-Tool-Wear-Prediction-LSTM/
├── toolwear.ipynb # Complete data processing, EDA, modeling notebook
├── app.py # Streamlit web application
├── Final_Best_model.h5 # Trained LSTM model
├── scaler.pkl # Feature scaler
├── requirements.txt # Project dependencies
└── README.md # Project documentation

---

## 🧪 Model Details
- **Model Type:** LSTM (Long Short-Term Memory)
- **Optimizer:** Adam
- **Loss Function:** Binary / Categorical Crossentropy
- **Reason for LSTM:** Handles temporal dependencies in sensor time-series data efficiently

---

## 🌐 Deployment
- **Platform:** AWS EC2 (Ubuntu)
- **Framework:** Streamlit
- **Access:** Web-based UI for real-time predictions

---

## 👩‍💻 Author
Sai Vennela Yadavalli

---

## 📜 License
This project is created for learning, experimentation, and educational purposes.
