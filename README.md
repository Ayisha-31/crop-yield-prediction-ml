Crop Yield Prediction Project
About the Project

This project is a machine learning–based system developed to predict crop yield using historical agricultural and climate data. Crop yield depends on several factors such as rainfall, temperature, pesticide usage, crop type, year, and region. By analyzing these factors together, the model learns patterns from past data and predicts future crop yield.

The main goal of this project is not just prediction accuracy, but understanding how real-world data is processed, modeled, evaluated, and interpreted using machine learning techniques.

Problem Statement

To predict crop yield based on historical agricultural and environmental data using machine learning models.

Dataset Used

The dataset used in this project is a public agricultural dataset containing more than 28,000 records.
Each record includes:

Area (Country/Region)

Crop type (Item)

Year

Average rainfall (mm/year)

Average temperature (°C)

Pesticide usage (tonnes)

Crop yield (hg/ha)

This dataset represents real-world agricultural conditions and is suitable for building predictive ML models.

Approach Used

The project follows a standard machine learning workflow:

Data Preprocessing

Removed missing and invalid values

Encoded categorical features like Area and Crop

Scaled numerical features for better model performance

Model Training

Trained multiple models such as Random Forest, Extra Trees, and Gradient Boosting

Used ensemble techniques to improve prediction stability

Selected the best-performing model based on evaluation metrics

Model Evaluation

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Prediction

The trained model predicts crop yield for new input values

Output is shown in both hg/ha and tonnes per hectare

Visualization

Scatter plots for rainfall vs yield and temperature vs yield

Correlation heatmap to understand feature relationships

Actual vs predicted yield comparison

Tools & Technologies

Python

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

VS Code

How to Run
python train_model.py
python predict.py
python visualization.py

Results

The Random Forest model produced stable and reliable predictions with low error values and a good R² score. The results show that rainfall and temperature have a noticeable impact on crop yield, and the model is able to capture these relationships effectively.

Conclusion

This project demonstrates how machine learning can be applied to real agricultural data to solve a practical prediction problem. It helped in understanding data preprocessing, model selection, evaluation, and result interpretation in a real-world scenario.

Future Work

Build a real-time dashboard using Streamlit

Integrate live weather data

Extend the model for region-specific predictions
