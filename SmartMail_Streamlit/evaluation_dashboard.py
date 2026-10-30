
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import multilabel_confusion_matrix

st.set_page_config(
    page_title="SmartMail Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Model Evaluation")
st.write("Performance comparison of four machine learning algorithms.")

# Load comparison results
results = pd.read_csv("model_comparison.csv")

# Model comparison table
st.subheader("Model Comparison")
st.dataframe(
    results.style.format({
        "Exact Match Accuracy": "{:.2%}",
        "Micro Precision": "{:.2%}",
        "Micro Recall": "{:.2%}",
        "Micro F1": "{:.2%}",
        "Macro F1": "{:.2%}",
        "Weighted F1": "{:.2%}",
        "Hamming Loss": "{:.2%}"
    }),
    use_container_width=True
)

# Best model
best = results.sort_values(
    "Micro F1",
    ascending=False
).iloc[0]

st.success(
    f"Best Model: {best['Model']} | "
    f"Micro F1: {best['Micro F1']:.2%}"
)

# Model comparison chart
st.subheader("Model Performance Comparison")

fig = px.bar(
    results,
    x="Model",
    y=["Micro F1", "Macro F1", "Weighted F1"],
    barmode="group",
    title="F1 Score Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# Classification report
st.markdown("---")
st.subheader("Category-wise Classification Report")

try:
    with open("classification_report.txt", "r") as file:
        report = file.read()

    st.code(report, language="text")

except FileNotFoundError:
    st.warning("Classification report not found.")

# Confusion matrix
st.markdown("---")
st.subheader("Confusion Matrix")

try:
    df = pd.read_csv("emails_dataset.csv")

    import ast
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    import re

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    X = (df["subject"] + " " + df["body"]).apply(clean_text)
    labels = df["labels"].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = joblib.load("Models/email_classifier_best.pkl")
    predictions = model.predict(X_test)

    matrices = multilabel_confusion_matrix(y_test, predictions)

    category = st.selectbox(
        "Select Email Category",
        mlb.classes_
    )

    index = list(mlb.classes_).index(category)
    matrix = matrices[index]

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"]
    )

    st.dataframe(matrix_df, use_container_width=True)

    fig_cm = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        title=f"Confusion Matrix - {category}",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

except Exception as e:
    st.error(f"Could not generate confusion matrix: {e}")
