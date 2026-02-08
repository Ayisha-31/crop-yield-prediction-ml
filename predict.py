"""
SIMPLE PREDICTION SCRIPT

"""

import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CROP YIELD PREDICTION - PREDICTION SYSTEM")
print("="*60)


print("\n[STEP 1] Loading trained model...")

try:
    with open('models/crop_yield_model.pkl', 'rb') as f:
        model_package = pickle.load(f)

    models = model_package['models']
    best_model_name = model_package['best_model']
    encoders = model_package['encoders']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']

    print(f"✓ Loaded model: {best_model_name}")
    print(f"✓ Training date: {model_package['training_date']}")

except FileNotFoundError:
    print("❌ ERROR: Model file not found!")
    print("Run: python train_model.py")
    exit()


print("\n[STEP 2] Preparing prediction inputs...")

predictions_data = [
    {
        'Area': 'India',
        'Item': 'Rice, paddy',
        'Year': 2024,
        'average_rain_fall_mm_per_year': 1200.0,
        'pesticides_tonnes': 50.0,
        'avg_temp': 28.0
    }
]

print(f"✓ Prepared {len(predictions_data)} request(s)")


print("\n[STEP 3] Making predictions...")
print("="*60)

results = []
best_model = models[best_model_name]

for i, input_data in enumerate(predictions_data, 1):
    print(f"\nPrediction #{i}:")
    
    input_df = pd.DataFrame([input_data])

    # ---------- Encode categorical safely ----------
    for col in ['Area', 'Item']:
        known_classes = encoders[col].classes_
        
        if input_df[col].iloc[0] not in known_classes:
            print(f"❌ ERROR: '{input_df[col].iloc[0]}' not seen during training for {col}")
            print(f"   Known values: {list(known_classes)[:10]} ...")
            print("   👉 Retrain model OR use known values only\n")
            break
        else:
            input_df[col] = encoders[col].transform(input_df[col])

    else:
      
        numerical_cols = [
            'Year',
            'average_rain_fall_mm_per_year',
            'pesticides_tonnes',
            'avg_temp'
        ]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = best_model.predict(input_df)[0]

        print(f"  ✓ Predicted Yield: {prediction:.2f} hg/ha")
        print(f"(~ {prediction/10000:.2f} tonnes/hectare)")


        results.append({
            **input_data,
            'Predicted_Yield_hg_ha': round(prediction, 2),
            'Predicted_Yield_tonnes_ha': round(prediction / 10000, 2)
        })



print("\n" + "="*60)
print("[STEP 4] Saving predictions...")

results_df = pd.DataFrame(results)
results_df.to_csv("predictions.csv", index=False)

print("✓ Saved predictions.csv")


print("\n" + "="*60)
print("PREDICTION COMPLETE")
print("="*60)
print(results_df.to_string(index=False))

