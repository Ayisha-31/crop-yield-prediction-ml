


import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    VotingRegressor
   
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


warnings.filterwarnings('ignore')

print("="*60)
print("CROP YIELD PREDICTION - TRAINING SYSTEM")
print("="*60)

# STEP 1: LOAD YOUR DATA
print("\n[STEP 1] Loading data...")

try:
    global_df = pd.read_csv("yield_df.csv")
    print(f"✓ Loaded global data: {global_df.shape}")
except:
    print("⚠ Could not find yield_df.csv")
    print("SOLUTION: Either:")
    print("  1. Download dataset from Kaggle: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset")
    print("  2. Or use the sample data generator below")
    
    print("\n→ Generating sample data for demonstration...")
    np.random.seed(42)
    global_df = pd.DataFrame({
        'Area': np.random.choice(['India', 'China', 'USA', 'Brazil'], 1000),
        'Item': np.random.choice(['Rice', 'Wheat', 'Maize', 'Soybeans'], 1000),
        'Year': np.random.randint(2000, 2024, 1000),
        'average_rain_fall_mm_per_year': np.random.uniform(400, 2500, 1000),
        'pesticides_tonnes': np.random.uniform(10, 200, 1000),
        'avg_temp': np.random.uniform(15, 35, 1000),
        'hg/ha_yield': np.random.uniform(1000, 5000, 1000)
    })
    print(f"✓ Generated sample data: {global_df.shape}")
if 'Unnamed: 0' in global_df.columns:
    global_df = global_df.drop(columns=['Unnamed: 0'])
    print("✓ Dropped unwanted column: 'Unnamed: 0'")

# STEP 2: PREPROCESS DATA
print("\n[STEP 2] Preprocessing data...")

# Remove missing values
global_df = global_df.dropna()
print(f"✓ After removing NaN: {global_df.shape}")

# Encode categorical variables 
encoders = {}
for col in ['Area', 'Item']:
    encoders[col] = LabelEncoder()
    global_df[col] = encoders[col].fit_transform(global_df[col])
print("✓ Encoded categorical variables")

# Separate features and target
target_col = 'hg/ha_yield'
X = global_df.drop(target_col, axis=1)
y = global_df[target_col]
print(f"✓ Features: {X.columns.tolist()}")
print(f"✓ Target: {target_col}")


numerical_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
scaler = RobustScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("✓ Scaled numerical features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")


print("\n[STEP 3] Building models (RAM-safe)...")

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=1
    ),

    'Extra Trees': ExtraTreesRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=1
    ),

    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),

    'XGBoost': XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=1,
        verbosity=0
    ),

    'LightGBM': LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=1,
        verbose=-1
    ),
}

print(f"✓ Created {len(models)} base models")

print("\n[STEP 4] Creating Voting Ensemble...")

models['Voting Ensemble'] = VotingRegressor(
    estimators=[
        ('rf', models['Random Forest']),
        ('et', models['Extra Trees']),
        ('xgb', models['XGBoost'])
    ]
)



print(f"✓ Total models: {len(models)}")


# Hyperparameter space for Random Forest
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

from sklearn.model_selection import RandomizedSearchCV

print("\n[HYPERPARAMETER TUNING] Random Forest...")

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=1
)

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='r2',
    cv=3,
    random_state=42,
    n_jobs=1,
    verbose=1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("✓ Best RF Params:", rf_search.best_params_)
print("✓ Best CV R²:", rf_search.best_score_)


models['Random Forest'] = best_rf


print(f"✓ Total models: {len(models)}")

# STEP 5: TRAIN ALL MODELS

print("\n[STEP 5] Training models (this may take 2-3 minutes)...")

results = []

for name, model in models.items():
    print(f"\n  Training {name}...", end=' ')
    

    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    
  
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })
    
    print(f"✓ (RMSE: {rmse:.2f}, R²: {r2:.4f})")


print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)

print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {results_df.iloc[0]['R²']:.4f}")
print(f"   RMSE: {results_df.iloc[0]['RMSE']:.2f}")

# STEP 7: SAVE MODELS
print("\n[STEP 7] Saving models...")


import os
os.makedirs('models', exist_ok=True)


model_package = {
    'models': models,
    'best_model': best_model_name,
    'encoders': encoders,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'target_name': target_col,
    'performance': results_df.to_dict('records'),
    'training_date': datetime.now().isoformat()
}

with open('models/crop_yield_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✓ Models saved to: models/crop_yield_model.pkl")

results_df.to_csv('models/model_performance.csv', index=False)
print("✓ Results saved to: models/model_performance.csv")

# STEP 8: TEST PREDICTION
print("\n[STEP 8] Testing prediction with sample data...")


sample_input = X_train.iloc[[0]].copy()

print("\nSample Input (from training data):")
print(sample_input)

best_model = models[best_model_name]
prediction = best_model.predict(sample_input)

print(f"\n✓ Predicted Yield: {prediction[0]:.2f} hg/ha")

# DONE!
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nWhat was created:")
print("  📁 models/crop_yield_model.pkl     - Saved model")
print("  📁 models/model_performance.csv    - Performance metrics")
print("\nNext steps:")
print("="*60)
