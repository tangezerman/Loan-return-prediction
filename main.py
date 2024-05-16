import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import xgboost as xgb

# Load the dataset
data = pd.read_csv('balanced_loan_data.csv')

# Data preprocessing
# Assuming 'purpose' was already one-hot encoded

# Prepare the features and target variable
# Assuming 'not.fully.paid' is the target
X = data.drop(['not.fully.paid'], axis=1)
y = data['not.fully.paid']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Initialize MLflow and set an experiment
mlflow.set_tracking_uri('http://127.0.0.1:8080')
mlflow.set_experiment('Loan Repayment Prediction')

# Start an MLflow run
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0] * len(y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log model, parameters, and metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
            registered_model_name=f"{name}_model"
        )

        # Output the results
        print(f"Run: {name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")

# Check results
print("Models and metrics logged successfully.")
