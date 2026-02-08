"""
CROP YIELD PREDICTION - VISUALIZATION SYSTEM

"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


print("="*60)
print("CROP YIELD PREDICTION - VISUALIZATION SYSTEM")
print("="*60)

sns.set(style="whitegrid")
os.makedirs("visualizations", exist_ok=True)


df = pd.read_csv("yield_df.csv")

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df.dropna(inplace=True)

# LOAD MODEL PACKAGE
with open("models/crop_yield_model.pkl", "rb") as f:
    model_package = pickle.load(f)

models = model_package["models"]
best_model_name = model_package["best_model"]
encoders = model_package["encoders"]
scaler = model_package["scaler"]

best_model = models[best_model_name]


df_encoded = df.copy()

for col in ["Area", "Item"]:
    df_encoded[col] = encoders[col].transform(df_encoded[col])

num_cols = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

X = df_encoded.drop("hg/ha_yield", axis=1)
y = df_encoded["hg/ha_yield"]

predictions = best_model.predict(X)
residuals = y - predictions

# Scatter: Temperature vs Yield
plt.figure(figsize=(7,5))
sns.scatterplot(
    x=df["avg_temp"],
    y=df["hg/ha_yield"],
    color="#FF6F61",
    alpha=0.6
)
plt.title("Temperature vs Crop Yield")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Yield (hg/ha)")
plt.tight_layout()
plt.savefig("visualizations/temp_vs_yield.png")
plt.close()

#  Scatter: Rainfall vs Yield
plt.figure(figsize=(7,5))
sns.scatterplot(
    x=df["average_rain_fall_mm_per_year"],
    y=df["hg/ha_yield"],
    color="#4A90E2",
    alpha=0.6
)
plt.title("Rainfall vs Crop Yield")
plt.xlabel("Rainfall (mm/year)")
plt.ylabel("Yield (hg/ha)")
plt.tight_layout()
plt.savefig("visualizations/rainfall_vs_yield.png")
plt.close()

#  Heatmap: Feature Correlation
plt.figure(figsize=(8,6))
corr = df[
    [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "hg/ha_yield"
    ]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.close()

#  Actual vs Predicted Yield
plt.figure(figsize=(7,5))
plt.scatter(y, predictions, color="#2ECC71", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
plt.xlabel("Actual Yield (hg/ha)")
plt.ylabel("Predicted Yield (hg/ha)")
plt.title("Actual vs Predicted Yield")
plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.close()

# Residual Plot
plt.figure(figsize=(7,5))
plt.scatter(predictions, residuals, color="#9B59B6", alpha=0.6)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted Yield (hg/ha)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("visualizations/residual_plot.png")
plt.close()

# Crop-wise Yield Distribution
top_crops = df["Item"].value_counts().head(6).index
df_top = df[df["Item"].isin(top_crops)]

plt.figure(figsize=(10,5))
sns.boxplot(
    x="Item",
    y="hg/ha_yield",
    data=df_top,
    palette="Set2"
)
plt.title("Crop-wise Yield Distribution")
plt.xlabel("Crop")
plt.ylabel("Yield (hg/ha)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("visualizations/crop_yield_distribution.png")
plt.close()

print(" VISUALIZATIONS GENERATED SUCCESSFULLY")
print(" Files saved in: visualizations/")
print("="*60)


