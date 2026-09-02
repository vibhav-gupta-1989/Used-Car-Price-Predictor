# Used Car Price Predictor

A machine learning regression project for predicting used-car selling prices using vehicle attributes such as age, mileage, fuel type, seller type, transmission, ownership history, and vehicle name.

The project demonstrates an end-to-end ML workflow including **EDA, feature engineering, preprocessing pipelines, model comparison, cross-validation, hyperparameter tuning, feature importance analysis, and final test evaluation**.

## Dataset

The project uses the **CarDekho Vehicle Dataset**, containing:

* **4,340 vehicles**
* **8 original attributes**
* Target: `selling_price`

Key features include:

`name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`.

## Machine Learning Pipeline

```text
CarDekho Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train/Test Split (80/20)
      │
      ▼
Feature Engineering
      │
      ├── Vehicle Age
      ├── Log-transformed Mileage
      └── Vehicle Name Decomposition
      │
      ▼
Preprocessing Pipeline
      │
      ├── StandardScaler
      ├── OneHotEncoder
      └── ColumnTransformer
      │
      ▼
Model Comparison
      │
      ├── Linear Regression
      ├── Decision Tree
      └── Random Forest
      │
      ▼
10-Fold Cross-Validation
      │
      ▼
Grid Search + Randomized Search
      │
      ▼
Feature Importance Analysis
      │
      ▼
Final Test Evaluation
```

## Feature Engineering

Several transformations were applied:

* Converted manufacturing `year` into **vehicle age**
* Applied a **log transformation** to the right-skewed `km_driven` feature
* Decomposed `name` into the **manufacturer** and remaining vehicle name
* One-hot encoded categorical features
* Standardized numerical features
* Applied `log1p` transformation to the target variable

The preprocessing pipeline transformed the **3,472 training samples into 1,368 features**.

## Models

Three regression models were evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

Random Forest was further optimized using **10-fold cross-validation**, Grid Search, and Randomized Search.

The best Randomized Search configuration used:

```text
max_features = 16
```

## Final Results

The tuned Random Forest achieved the following performance on the unseen test set:

| Metric   |       Result |
| -------- | -----------: |
| **RMSE** | **₹320,973** |
| **MAE**  | **₹104,166** |
| **R²**   |   **0.6624** |

Compared with a `DummyRegressor` baseline:

| Model             |         RMSE |          MAE |
| ----------------- | -----------: | -----------: |
| Dummy Regressor   |     ₹552,722 |     ₹313,630 |
| **Random Forest** | **₹320,973** | **₹104,166** |

This represents approximately:

* **42% lower RMSE**
* **67% lower MAE**

than the mean-value baseline.

## Feature Importance

The most important features included:

1. **Vehicle age** — 18.4%
2. **Log-transformed mileage** — 8.0%
3. **Diesel fuel** — 5.0%
4. **Automatic transmission** — 4.3%
5. **Manual transmission** — 4.0%

Vehicle age was the strongest predictor of selling price.

## Technologies

* Python 3.12
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Matplotlib
* KaggleHub
* Jupyter Notebook

## Key Concepts

* Regression
* Exploratory Data Analysis
* Feature Engineering
* Data Preprocessing
* Scikit-learn Pipelines
* ColumnTransformer
* Target Transformation
* Cross-Validation
* Grid Search
* Randomized Search
* Random Forest
* Feature Importance
* RMSE, MAE, and R²

## Project Structure

```text
Used-Car-Price-Predictor/
│
├── Car Price Predictor.ipynb
└── README.md
```

## Running the Project

```bash
git clone <your-repository-url>
cd Used-Car-Price-Predictor
pip install numpy pandas matplotlib scikit-learn scipy kagglehub jupyter
jupyter notebook
```

Open `Car Price Predictor.ipynb` and run the cells sequentially.

## Future Improvements

* Gradient-boosting models such as XGBoost, LightGBM, or CatBoost
* Improved handling of high-cardinality vehicle names
* More extensive hyperparameter optimization
* Residual and error analysis
* Model deployment through a REST API or web application
