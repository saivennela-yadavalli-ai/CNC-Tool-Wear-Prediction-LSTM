import streamlit as st
import base64
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tool Wear Prediction",
    page_icon="🔧",
    layout="wide"
)

# --------------------------------------------------
# Background image function
# --------------------------------------------------
def set_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background-color: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👉 Update path if needed
set_bg(r"C:\Sai\Guvi Capstone Projects\Project\BG Pic.png")

# --------------------------------------------------
# GLOBAL CSS (UI + VISIBILITY FIX)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* General text */
    h1, h2, h3, p, label {
        color: #ffffff !important;
    }

    /* Main content card */
    .block-container {
        background-color: rgba(0, 0, 0, 0.55);
        border-radius: 12px;
        padding: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
    }

    [data-testid="stSidebar"] .block-container {
        background-color: #1e1e1e !important;
    }

    /* Sidebar labels */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffd54f !important;
        font-weight: 600;
    }

    /* ALL sidebar number inputs */
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #333333 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #555555 !important;
    }

    [data-testid="stSidebar"] .stNumberInput input:focus {
        border: 1px solid #ffd54f !important;
        outline: none !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background-color: #444444 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }

    /* Home page card */
    .home-card {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 30px;
        border-radius: 14px;
        margin-top: 20px;
    }

    .home-card ul li {
        color: #ffffff;
        font-size: 17px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load model & scaler
# --------------------------------------------------
@st.cache_resource
def load_resources(model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_resources(
    r"C:\Sai\Guvi Capstone Projects\Project\Final_Best_model.h5",
    r"C:\Sai\Guvi Capstone Projects\Project\scaler.pkl"
)

# --------------------------------------------------
# Feature list (order matters)
# --------------------------------------------------
selected_columns = [
    "material",
    "feedrate",
    "clamp_pressure",
    "X1_ActualPosition",
    "Y1_ActualPosition",
    "Z1_ActualPosition",
    "X1_CurrentFeedback",
    "Y1_CurrentFeedback",
    "M1_CURRENT_FEEDRATE",
    "X1_DCBusVoltage",
    "X1_OutputPower",
    "Y1_OutputPower",
    "S1_OutputPower"
]

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
menu = ["Home", "Prediction"]
choice = st.sidebar.selectbox("Navigation", menu)

# --------------------------------------------------
# HOME PAGE (FIXED VISIBILITY)
# --------------------------------------------------
if choice == "Home":
    st.markdown(
        """
        <div class="home-card">
            <h2>🔬 Tool Wear Prediction using LSTM</h2>
            <p>This application predicts:</p>
            <ul>
                <li>Tool Condition (Good / Worn / Damaged)</li>
                <li>Machining Finalization Status</li>
                <li>Visual Inspection Outcome</li>
            </ul>
            <p>
                Please navigate to <b>Prediction</b> and enter the machining parameters.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
else:
    st.title("🧪 Tool Wear Prediction")
    st.sidebar.header("📥 Input Parameters")

    user_input = {
        "material": st.sidebar.number_input("material", value=0),
        "feedrate": st.sidebar.number_input("feedrate", value=0.0),
        "clamp_pressure": st.sidebar.number_input("clamp_pressure", value=0.0),
    }

    for feat in selected_columns[3:]:
        user_input[feat] = st.sidebar.number_input(feat, value=0.0)

    df_input = pd.DataFrame([user_input])
    scaled_input = scaler.transform(df_input)

    # LSTM input shape: (samples, timesteps, features)
    sequence = np.tile(scaled_input, (10, 1)).reshape(1, 10, len(selected_columns))

    if st.sidebar.button("Predict"):
        with st.spinner("Predicting..."):
            tool_cond, machining_flag, visual_flag = model.predict(sequence)

        tool_labels = {0: "Good", 1: "Worn", 2: "Damaged"}

        st.subheader("🔍 Prediction Results")

        st.write(f"**🛠 Tool Condition:** {tool_labels[np.argmax(tool_cond[0])]}")

        st.write(
            f"**🔄 Machining Finalized:** "
            f"{'✅ Yes' if machining_flag[0][0] > 0.5 else '❌ No'}"
        )

        st.write(
            f"**👀 Visual Inspection Passed:** "
            f"{'✅ Yes' if visual_flag[0][0] > 0.5 else '❌ No'}"
        )

        st.success("✅ Prediction Completed Successfully")
