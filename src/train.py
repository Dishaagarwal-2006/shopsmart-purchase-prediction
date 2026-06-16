# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# 1. Load Data
df = pd.read_csv("dataset/shop_smart_ecommerce.csv")
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define Pipelines (Using your optimal grid parameters)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Using optimal hyperparameters found from your grid search
dt_model = DecisionTreeClassifier(
    max_depth=4, 
    min_samples_leaf=50, 
    class_weight="balanced", 
    random_state=42
)

# Fit pipeline
preprocessor.fit(X_train)
X_train_trans = preprocessor.transform(X_train)
dt_model.fit(X_train_trans, y_train)

# 3. Export Artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
joblib.dump(dt_model, "artifacts/model.pkl")
print("Model and Preprocessor exported successfully to /artifacts!")