
import pandas as pd
import matplotlib.pyplot as plt

from importlib.resources import files

from paradigma.config import IMUConfig, TremorConfig
from paradigma.constants import DataColumns
from paradigma.preprocessing import preprocess_imu_data

from paradigma.pipelines.tremor_pipeline import (
    extract_tremor_features,
    detect_tremor,
)


# ==========================================
# 1. LOAD YOUR DATA
# ==========================================

df = pd.read_csv(
    "001_CrossArms_LeftWrist.txt",
    header=None,
    names=[
        "time",
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    ],
)
sample_interval = df["time"].diff().median()
df["data_segment_nr"] = (
    df["time"].diff().fillna(sample_interval) > 3 * sample_interval
).cumsum()

# Tell ParaDigMa which columns contain which measurements.
column_mapping = {}

# IMPORTANT:
# This example assumes gyroscope measurements are in degrees/second.
# Convert them first if your dataset uses radians/second.

imu_config = IMUConfig(column_mapping)


# ==========================================
# 2. PREPARE THE SENSOR DATA
# ==========================================

print("Preparing sensor data...")

prepared_data = preprocess_imu_data(
    df,
    imu_config,
    sensor="gyroscope",
    watch_side="left"
)

print("Data preparation complete!")


# ==========================================
# 3. ANALYZE POSSIBLE TREMORS
# ==========================================

print("Analyzing hand movements...")

tremor_config = TremorConfig()

tremor_config._set_sampling_frequency_detected(
    imu_config.sampling_frequency
)

# Extract information about the shaking.
features = extract_tremor_features(
    prepared_data,
    tremor_config
)

# Load the existing trained tremor detector.
classifier = (
    files("paradigma")
    / "assets"
    / "tremor_detection_clf_package.pkl"
)

# Ask the existing detector to analyze the data.
results = detect_tremor(
    features,
    tremor_config,
    classifier
)

print("Tremor analysis complete!")


# ==========================================
# 4. DISPLAY THE RESULTS
# ==========================================

result_columns = [
    tremor_config.time_colname,
    DataColumns.PRED_TREMOR_PROBA,
    DataColumns.PRED_TREMOR_LOGREG,
    DataColumns.PRED_TREMOR_CHECKED,
    DataColumns.TREMOR_POWER,
]

print("\nAUTOME RESULTS")

print(results[result_columns].head(20))

# Save the results.
results[result_columns].to_csv(
    "autome_results.csv",
    index=False
)

print("\nResults saved to autome_results.csv")


# ==========================================
# 5. DRAW A SIMPLE GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    results[tremor_config.time_colname],
    results[DataColumns.PRED_TREMOR_PROBA],
    label="Tremor probability"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Predicted tremor probability")
plt.title("AutoMe - Tremor Analysis")

plt.legend()
plt.grid(True)

plt.savefig("autome_graph.png")
plt.show()
