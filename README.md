# Interactive Corrosion Rate Predictor 🧪

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)


An interactive web application that uses a machine learning model to predict the corrosion rate of a metal component based on its environmental and operational conditions. This tool is designed for materials engineers, maintenance teams, and students in corrosion engineering.

## Demo


*The user interface allows for real-time adjustments of parameters to see an instant prediction of the corrosion rate.*

---

## 📋 Key Features

* **Real-Time Prediction:** Instantly calculates the corrosion rate (mm/year) as you adjust input parameters.
* **Interactive UI:** User-friendly sliders and number inputs for temperature, pH, chloride concentration, and flow velocity.
* **Risk Assessment:** Provides a simple risk classification (Low, Moderate, High) based on the predicted rate.
* **Modular Codebase:** The project is structured with separate scripts for data generation, model training, and the web application.

---

## 🛠️ Technology Stack

* **Backend:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Web Framework:** Streamlit
* **Model Serialization:** Joblib

---

## 📁 Project Structure

```
corrosion_predictor_app/
│
├── data/                  # Stores the generated dataset
│   └── corrosion_data.csv
│
├── models/                # Stores the trained model and scaler
│   ├── corrosion_model.joblib
│   └── scaler.joblib
│
├── src/                   # Source code for the ML pipeline
│   ├── __init__.py
│   ├── generate_data.py   # Script to create the synthetic dataset
│   └── train.py           # Script to train the ML model
│
├── app.py                 # The main Streamlit web application file
└── requirements.txt       # Project dependencies
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.9 or higher installed on your system.

### Step-by-Step Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/corrosion-rate-predictor-app.git](https://github.com/5h4h1d-k/corrosion-rate-predictor-app.git)
    cd corrosion-predictor-app
    ```

2.  **Create and activate a virtual environment:**
    A virtual environment is a self-contained directory that holds all the necessary packages for a project, keeping it isolated from your other Python projects.

    * **Windows (Command Prompt):**
        ```cmd
        python -m venv venv
        venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    You will know it's active when you see `(venv)` at the beginning of your terminal prompt.

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the one-time ML pipeline scripts:**
    These scripts need to be run only once to create the data and train the model.

    * First, generate the synthetic training data:
        ```bash
        python src/generate_data.py
        ```
    * Next, train the model using this data:
        ```bash
        python src/train.py
        ```

5.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will now be running and accessible in your web browser, typically at `http://localhost:8501`.

---

## 🔮 Future Improvements

* Integrate a real-world corrosion dataset instead of synthetic data.
* Add more input features (e.g., material composition, pressure, presence of inhibitors).
* Deploy the application to a cloud service like Streamlit Community Cloud or Heroku.
* Incorporate data visualization to show the impact of each parameter on the corrosion rate.

---

