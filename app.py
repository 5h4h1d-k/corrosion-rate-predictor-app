# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Corrosion Rate Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Asset Loading ---
# Use a caching decorator to load the model and scaler only once, improving performance.
@st.cache_resource
def load_assets():
    """Loads the trained model and scaler from the 'models' directory."""
    model_path = os.path.join('models', 'corrosion_model.joblib')
    scaler_path = os.path.join('models', 'scaler.joblib')

    # Check if the model files exist before trying to load them.
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found! Please run the training script `src/train.py` first.")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# --- Application UI ---
st.title("⚙️ Interactive Corrosion Rate Predictor")
st.markdown(
    """
    This tool uses a machine learning model to estimate the corrosion rate of a metal component.
    Adjust the parameters in the sidebar to see the prediction in real-time.
    """
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Use the sliders to set the conditions.")

def get_user_input():
    """Creates sidebar widgets and returns user inputs as a DataFrame."""
    temp = st.sidebar.slider('Temperature (°C)', 20.0, 150.0, 85.0, 0.5)
    ph = st.sidebar.slider('pH of Fluid', 3.0, 11.0, 7.0, 0.1)
    chloride = st.sidebar.number_input('Chloride Concentration (ppm)', min_value=50, max_value=10000, value=5000, step=50)
    velocity = st.sidebar.slider('Flow Velocity (m/s)', 0.1, 5.0, 2.5, 0.1)

    data = {
        'Temperature_C': temp,
        'pH': ph,
        'Chloride_ppm': chloride,
        'Flow_Velocity_ms': velocity
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# --- Prediction and Display Logic ---
# Only show the main content if the model has been loaded successfully.
if model and scaler:
    with st.expander("View current input parameters", expanded=False):
        st.dataframe(input_df)

    # 1. Scale the user's input using the pre-fitted scaler.
    input_scaled = scaler.transform(input_df)

    # 2. Make a prediction.
    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result")

    # Create two columns
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("#### Current Settings")
        st.dataframe(input_df)

    with col2:
        st.metric(label="Predicted Corrosion Rate", value=f"{prediction[0]:.4f} mm/year")
        if prediction[0] < 0.2:
            st.success("✅ **Low Risk:** Standard monitoring is likely sufficient.")
        elif prediction[0] < 0.5:
            st.warning("⚠️ **Moderate Risk:** Increased monitoring is recommended.")
        else:
            st.error("🚨 **High Risk:** Action may be required to mitigate corrosion.")

    st.markdown("---")
    with st.expander("How does this work?"):
        st.markdown("""
        This tool is powered by a **Random Forest** model trained on 2,000 synthetic data points.
        When you adjust the sliders, your inputs are scaled and fed to the model to generate a live prediction.
        """)

import matplotlib.pyplot as plt
import seaborn as sns

# ... (inside the 'if model and scaler:' block)

st.markdown("---")
st.subheader("Which Factors Matter Most?")

# Get feature importances from the model
feature_importances = model.feature_importances_
features = ['Temperature', 'pH', 'Chloride', 'Velocity']

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
ax.set_title('Feature Importance for Corrosion Rate')
ax.set_xlabel('Importance')
ax.set_ylabel('')

# Display the plot in Streamlit
st.pyplot(fig)

with st.expander("How to interpret this chart"):
    st.write("""
        This chart shows the relative importance of each input parameter in the model's decision-making process.
        A higher importance score means that changes to that parameter will have a larger effect on the predicted corrosion rate.
    """)