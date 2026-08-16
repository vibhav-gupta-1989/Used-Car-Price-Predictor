# Used Car Price Predictor

- Developed a machine learning regression system to predict used-car selling prices using the CarDekho vehicle dataset (4,340 vehicles).
- Built an end-to-end Scikit-learn pipeline for preprocessing and feature engineering.
- Extracted car manufacturer information from vehicle names and engineered car age from the manufacturing year.
- Applied log transformation and standardization to numerical features and one-hot encoding to categorical features.
- Evaluated Linear Regression, Decision Tree, and Random Forest regressors using 10-fold cross-validation.
- Applied log transformation to the target variable using TransformedTargetRegressor.
- Tuned Random Forest hyperparameters using Grid Search and Randomized Search.
- Achieved MAE of ₹1.04 lakh, RMSE of ₹3.21 lakh, and R² of 0.662 on the held-out test set.
- Improved MAE by 67% and RMSE by 42% over a mean-price dummy-regressor baseline.
