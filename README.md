# Used Car Price Predictor

- Developed an end-to-end machine learning pipeline to predict used car selling prices using Python and Scikit-learn, including data preprocessing, feature engineering, and model evaluation.
- Built modular preprocessing pipelines with ColumnTransformer, handling missing values, one-hot encoding of categorical features, feature scaling, and logarithmic transformations for skewed variables.
- Trained and compared Linear Regression, Decision Tree, and Random Forest regression models using 10-fold cross-validation, selecting the best model through hyperparameter tuning with GridSearchCV and RandomizedSearchCV.
- Implemented TransformedTargetRegressor to model log-transformed target values and evaluated model performance using RMSE on unseen test data.
- Performed feature importance analysis to identify the most influential predictors and guide model refinement.
