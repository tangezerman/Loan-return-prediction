import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import xgboost as xgb
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# List of CSV file paths
file_paths = ['batches/batch_1R.csv', 'batches/batch_2R.csv', 'batches/batch_3R.csv', 'batches/batch_4R.csv']  # Add your file paths here

# Initialize MLflow and set the tracking URI
mlflow.set_tracking_uri('http://127.0.0.1:8080')

# Define search spaces
search_spaces = {
    "Random_Forest": {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'min_samples_split': hp.quniform('min_samples_split', 2, 5, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1)
    },
    "Decision_Tree": {
        'max_depth': hp.choice('max_depth', [30, 40, 50]),
        'min_samples_split': hp.quniform('min_samples_split', 2, 5, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1)
    },
    "XGBoost": {       
        'max_depth': hp.choice('max_depth', [5, 6, 9]),
        'learning_rate': hp.uniform('learning_rate', 0.2, 0.6),
        'subsample': hp.choice('subsample', [0.5, 1]),
    }
}

# Define models
model_classes = {
    "Random_Forest": RandomForestClassifier,
    "Decision_Tree": DecisionTreeClassifier,
    "XGBoost": xgb.XGBClassifier
}

# Define the objective function for hyperopt
def objective(params, model_name, X_train, y_train, X_test, y_test):
    # Ensure min_samples_split and min_samples_leaf are integers if model is RandomForest or DecisionTree
    if model_name in ["Random_Forest", "Decision_Tree"]:
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

    # Print the parameters for debugging
    print(f"Parameters for {model_name}: {params}")

    model = model_classes[model_name](**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0] * len(y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    return {'loss': -roc_auc, 'status': STATUS_OK}

# Iterate over each file
for file_path in file_paths:
    # Load the dataset
    data = pd.read_csv(file_path)

    # Prepare the features and target variable

    X = data.drop(['not.fully.paid'], axis=1)
    y = data['not.fully.paid']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Start an MLflow run for each model
    for name in model_classes.keys():
        experiment_name = f"Loan Repayment Prediction - {os.path.basename(file_path)}"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{name}_default_{os.path.basename(file_path)}"):
            # Train the model with default parameters
            model = model_classes[name](random_state=42)
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

            # Log metrics
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
                registered_model_name=name + "_default"
            )

            # Output the results
            print(f"Default Run: {name} on {os.path.basename(file_path)}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"ROC AUC: {roc_auc}")

        with mlflow.start_run(run_name=f"{name}_optimized_{os.path.basename(file_path)}"):
            # Optimize hyperparameters
            trials = Trials()
            best_params = fmin(
                fn=lambda params: objective(params, name, X_train, y_train, X_test, y_test),
                space=search_spaces[name],
                algo=tpe.suggest,
                max_evals=25,
                trials=trials
            )

            # Ensure min_samples_split and min_samples_leaf are integers for RandomForest and DecisionTree
            if name in ["Random_Forest", "Decision_Tree"]:
                best_params['min_samples_split'] = int(best_params['min_samples_split'])
                best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

            # Print the best parameters for debugging
            print(f"Best parameters for {name}: {best_params}")

            # Validate max_depth for DecisionTree
            if name == "Decision_Tree" and best_params['max_depth'] <= 0:
                best_params['max_depth'] = None

            # Train the model with the best parameters
            model = model_classes[name](**best_params, random_state=42)
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

            # Log metrics
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
                registered_model_name=name
            )

            # Output the results
            print(f"Optimized Run: {name} on {os.path.basename(file_path)}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"ROC AUC: {roc_auc}")

# Check results
print("Models and metrics logged successfully.")
