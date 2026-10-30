
import pandas as pd
import ast
import joblib
import os
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv("emails_dataset.csv")

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

df["email_text"] = (
    df["subject"].astype(str) + " " +
    df["body"].astype(str)
)

df["labels"] = df["labels"].apply(ast.literal_eval)

# ==========================================
# 2. TEXT CLEANING
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_text"] = df["email_text"].apply(clean_text)

X = df["cleaned_text"]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

print("Total Emails:", len(df))
print("Total Categories:", len(mlb.classes_))
print("Categories:", mlb.classes_.tolist())

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training Emails:", len(X_train))
print("Testing Emails:", len(X_test))

# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    ),

    "Naive Bayes": OneVsRestClassifier(
        MultinomialNB()
    ),

    "Random Forest": OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    ),

    "XGBoost": OneVsRestClassifier(
        XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    )
}

# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

os.makedirs("Models", exist_ok=True)

for name, classifier in models.items():

    print("\n===================================")
    print("Training:", name)
    print("===================================")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )
        ),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Evaluation metrics
    exact_accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    hamming = hamming_loss(y_test, predictions)

    results.append({
        "Model": name,
        "Exact Match Accuracy": exact_accuracy,
        "Micro Precision": precision,
        "Micro Recall": recall,
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Hamming Loss": hamming
    })

    trained_models[name] = pipeline

    # Save individual trained model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        pipeline,
        f"Models/{filename}_trained.pkl"
    )

    print("Exact Match Accuracy:", round(exact_accuracy, 4))
    print("Micro Precision:", round(precision, 4))
    print("Micro Recall:", round(recall, 4))
    print("Micro F1:", round(micro_f1, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Hamming Loss:", round(hamming, 4))

# ==========================================
# 6. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Micro F1",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

# ==========================================
# 7. BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

joblib.dump(
    best_model,
    "Models/email_classifier_best.pkl"
)

joblib.dump(
    mlb,
    "Models/label_binarizer.pkl"
)

# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

best_predictions = best_model.predict(X_test)

report = classification_report(
    y_test,
    best_predictions,
    target_names=mlb.classes_,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

with open("classification_report.txt", "w") as file:
    file.write(report)

# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

confusion_matrices = multilabel_confusion_matrix(
    y_test,
    best_predictions
)

print("\n========== CONFUSION MATRICES ==========")

for category, matrix in zip(
    mlb.classes_,
    confusion_matrices
):
    print("\nCategory:", category)
    print(matrix)

print("\n========== TRAINING COMPLETED ==========")
print("Comparison saved: model_comparison.csv")
print("Report saved: classification_report.txt")
print("Best model saved: Models/email_classifier_best.pkl")
