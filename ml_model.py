import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

MODEL_PATH = "protein_xgb_model.pkl"

def train_model(csv_path="data/proteins.csv"):
    df = pd.read_csv(csv_path)
    X = df[["length", "hydrophobicity", "charge"]]
    y = df["enzymatic"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load_model():
    return joblib.load(MODEL_PATH)

def predict_protein(features: dict):
    """
    Predict enzyme/non-enzyme.
    features: {"length": int, "hydrophobicity": float, "charge": int}
    """
    model = load_model()
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return "Enzyme" if pred == 1 else "Non-Enzyme"

if __name__ == "__main__":
    train_model()
