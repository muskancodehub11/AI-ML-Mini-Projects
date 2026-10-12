import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print(df.head())

# 2. Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3. Remove customer ID
df = df.drop("customerID", axis=1)

# 4. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# 5. Find numerical and categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# 6. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 7. Create Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. TRAIN THE MODEL
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))




#  12: Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# 13: Define hyperparameters
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

# 14: Create GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# 15: Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# 16: Display best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

# 17: Display best cross-validation accuracy
print("Best CV Accuracy:", grid_search.best_score_)

# 18: Get the best model
best_model = grid_search.best_estimator_

# 19: Make predictions
y_pred_tuned = best_model.predict(X_test)

# 20: Evaluate tuned model
print("Tuned Test Accuracy:",
      accuracy_score(y_test, y_pred_tuned))

print("\nTuned Classification Report:")
print(classification_report(y_test, y_pred_tuned))
