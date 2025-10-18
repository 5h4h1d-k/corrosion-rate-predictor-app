
import pandas as pd
import numpy as np
import os

def generate_corrosion_data(num_samples=2000, save_path='data/corrosion_data.csv'):
    """
    Generates a synthetic dataset for corrosion rate prediction based on
    simplified, physics-informed principles and saves it to a CSV file.
    """
    print("Generating synthetic corrosion data...")
    np.random.seed(42) # Ensures that the data is the same every time we run it

    # Feature Generation: Simulating realistic operational conditions
    temperature = np.random.uniform(20, 150, num_samples)
    ph = np.random.uniform(3, 11, num_samples)
    chloride_ppm = np.random.uniform(50, 10000, num_samples)
    flow_velocity = np.random.uniform(0.1, 5.0, num_samples)

    # Target Calculation: A simplified model of corrosion
    # We combine the effects of each factor and add random "noise" to simulate real-world variability.
    base_rate = 0.05
    temp_effect = 0.001 * (temperature - 20)**1.1      # Non-linear temperature effect
    ph_effect = 0.02 * ((ph - 7)**2)                   # Corrosion is higher at acidic/alkaline extremes
    chloride_effect = 0.000025 * chloride_ppm**1.05  # Non-linear chloride effect
    flow_effect = 0.01 * np.sqrt(flow_velocity)
    noise = np.random.normal(0, 0.05, num_samples)     # Simulates measurement error

    corrosion_rate = base_rate + temp_effect + ph_effect + chloride_effect + flow_effect + noise
    corrosion_rate = np.maximum(0, corrosion_rate)     # Corrosion rate cannot be negative

    df = pd.DataFrame({
        'Temperature_C': temperature, 'pH': ph, 'Chloride_ppm': chloride_ppm,
        'Flow_Velocity_ms': flow_velocity, 'Corrosion_Rate_mm_per_year': corrosion_rate
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset with {num_samples} samples successfully created at '{save_path}'")

if __name__ == "__main__":
    generate_corrosion_data()