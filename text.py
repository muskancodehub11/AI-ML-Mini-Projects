"""
House Price Prediction: Cleaning -> EDA -> Modeling
Target: price (continuous)  =>  REGRESSION problem
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PATH = "Housing.csv"          # change if your file has a different name/location
OUT = "eda_plots/"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")


# 1. DATA CLEANING

df = pd.read_csv(PATH)
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# Standardise column names and text values
df.columns = df.columns.str.strip().str.lower()
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].str.strip().str.lower()

# Check outliers with the IQR rule
num_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
print("\nIQR outlier counts:")
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"  {c:10s}: {n}")

# Remove only EXTREME outliers (beyond 3*IQR) in price/area
def extreme(s):
    q1, q3 = s.quantile([.25, .75])
    i = q3 - q1
    return (s < q1 - 3 * i) | (s > q3 + 3 * i)

mask = extreme(df["price"]) | extreme(df["area"])
print("\nExtreme rows removed:", mask.sum())
df_clean = df[~mask].reset_index(drop=True)
print("Rows after cleaning:", len(df_clean))

# Encode categorical columns
bin_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea"]
for c in bin_cols:
    df_clean[c] = df_clean[c].map({"yes": 1, "no": 0})

df_enc = pd.get_dummies(df_clean, columns=["furnishingstatus"],
                        drop_first=True, dtype=int)
df_enc.to_csv("housing_cleaned.csv", index=False)


# 2. EDA

print("\n--- Price summary ---")
print(df_clean["price"].describe().round(0))
print("Price skewness:", round(df_clean["price"].skew(), 2))

# 2a. Price distribution (raw vs log)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_clean["price"], kde=True, ax=ax[0], color="#3b82f6")
ax[0].set_title("Price distribution")
sns.histplot(np.log1p(df_clean["price"]), kde=True, ax=ax[1], color="#10b981")
ax[1].set_title("log(Price) distribution")
plt.tight_layout(); plt.savefig(OUT + "01_price_distribution.png", dpi=130); plt.close()

# 2b. Correlation heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(df_enc.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, annot_kws={"size": 7})
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig(OUT + "02_correlation_heatmap.png", dpi=130); plt.close()

corr = df_enc.corr()["price"].drop("price").sort_values(ascending=False)
print("\n--- Correlation with price ---")
print(corr.round(3))

# 2c. Numeric features vs price
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, c in zip(ax.ravel(), ["area", "bedrooms", "bathrooms", "stories", "parking"]):
    if c == "area":
        sns.scatterplot(data=df_clean, x=c, y="price", ax=a, alpha=.6)
    else:
        sns.boxplot(data=df_clean, x=c, y="price", ax=a)
    a.set_title(f"Price vs {c}")
ax.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig(OUT + "03_numeric_vs_price.png", dpi=130); plt.close()

# 2d. Categorical features vs price
cat_cols = bin_cols + ["furnishingstatus"]
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for a, c in zip(ax.ravel(), cat_cols):
    sns.boxplot(data=df_clean, x=c, y="price", ax=a)
    a.set_title(f"Price vs {c}")
ax.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig(OUT + "04_categorical_vs_price.png", dpi=130); plt.close()


# 3. MODELING

X = df_enc.drop(columns="price")
y = df_enc["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    cv = cross_val_score(model, X, y, scoring="r2",
                         cv=KFold(5, shuffle=True, random_state=42)).mean()
    print(f"{name:30s} R2={r2:.3f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  CV-R2={cv:.3f}")
    return model, pred

print("\n--- Model comparison ---")
lr, p_lr = evaluate("Linear Regression (baseline)", LinearRegression())
rf, p_rf = evaluate("Random Forest (default)",
                    RandomForestRegressor(n_estimators=300, random_state=42))

# Hyperparameter tuning for Random Forest
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5, 1.0],
}
search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                            param_dist, n_iter=30, cv=5, scoring="r2",
                            random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
print("\nBest params:", search.best_params_)
best_rf, p_best = evaluate("Random Forest (tuned)", search.best_estimator_)

# Feature importance
imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values()
print("\n--- Feature importance ---")
print(imp.sort_values(ascending=False).round(3))

plt.figure(figsize=(8, 6))
imp.plot(kind="barh", color="#3b82f6")
plt.title("Random Forest feature importance")
plt.tight_layout(); plt.savefig(OUT + "05_feature_importance.png", dpi=130); plt.close()

# Actual vs predicted + residuals
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(y_test, p_best, alpha=.6)
lim = [y_test.min(), y_test.max()]
ax[0].plot(lim, lim, "r--")
ax[0].set_xlabel("Actual price"); ax[0].set_ylabel("Predicted price")
ax[0].set_title("Actual vs Predicted (tuned RF)")
sns.histplot(y_test - p_best, kde=True, ax=ax[1])
ax[1].set_title("Residuals")
plt.tight_layout(); plt.savefig(OUT + "06_actual_vs_predicted.png", dpi=130); plt.close()

# Save model
joblib.dump(best_rf, "house_price_rf_model.pkl")


# 4. PREDICT A NEW HOUSE

new_house = pd.DataFrame([{
    "area": 6000, "bedrooms": 3, "bathrooms": 2, "stories": 2,
    "mainroad": 1, "guestroom": 0, "basement": 1, "hotwaterheating": 0,
    "airconditioning": 1, "parking": 2, "prefarea": 1,
    "furnishingstatus_semi-furnished": 1, "furnishingstatus_unfurnished": 0,
}])[X.columns]   # keep the same column order as training
print(f"\nPredicted price for new house: {best_rf.predict(new_house)[0]:,.0f}")