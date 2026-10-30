import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import ast
import re

from pathlib import Path
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartMail | AI Email Classifier",
    page_icon="📧",
    layout="wide"
)

BASE_DIR = Path(__file__).parent


# --------------------------------------------------
# PREMIUM DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #080d1b 0%, #10172a 55%, #11152b 100%);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid #26334d;
}

[data-testid="stSidebar"] h1 {
    color: #60a5fa;
    font-weight: 800;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #cbd5e1;
}

.main .block-container {
    padding: 2.5rem 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243b, #111b30);
    border: 1px solid #2b4163;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border: none;
    box-shadow: 0 5px 20px rgba(59,130,246,0.35);
}

[data-testid="stDataFrame"] {
    border: 1px solid #2b4163;
    border-radius: 12px;
    overflow: hidden;
}

input, textarea {
    background-color: #17243b !important;
    color: white !important;
    border: 1px solid #344767 !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] > div {
    background-color: #17243b;
    color: white;
    border-color: #344767;
}

hr {
    border-color: #26334d;
}

.stAlert {
    border-radius: 12px;
}

[data-testid="stPlotlyChart"] {
    background: #111b30;
    padding: 12px;
    border: 1px solid #263b5b;
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():

    return pd.read_csv(
        BASE_DIR / "emails_dataset.csv"
    )


# --------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        BASE_DIR / "Models" / "email_classifier_best.pkl"
    )

    label_binarizer = joblib.load(
        BASE_DIR / "Models" / "label_binarizer.pkl"
    )

    return model, label_binarizer


df = load_dataset()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📧 SmartMail")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Explorer",
        "Email Prediction",
        "Analytics",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-Based Email Classification System"
)


# ==================================================
# 1. DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("📧 SmartMail Dashboard")

    st.write(
        "AI-Based Email Classification System"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Emails",
        len(df)
    )

    col2.metric(
        "Total Features",
        len(df.columns)
    )

    col3.metric(
        "Total Categories",
        10
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.success(
        "SmartMail dashboard is ready!"
    )


# ==================================================
# 2. DATASET EXPLORER
# ==================================================

elif page == "Dataset Explorer":

    st.title("📂 Dataset Explorer")

    uploaded_file = st.file_uploader(
        "Upload another CSV dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

    else:

        data = df

    st.metric(
        "Total Records",
        len(data)
    )

    st.subheader("Dataset")

    st.dataframe(
        data,
        width="stretch"
    )

    st.markdown("---")

    st.subheader("Missing Values")

    missing = data.isnull().sum().reset_index()

    missing.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing,
        width="stretch"
    )


# ==================================================
# 3. EMAIL PREDICTION WITH CONFIDENCE SCORE
# ==================================================

elif page == "Email Prediction":

    st.title("🤖 Email Classification")

    st.write(
        "Enter an email subject and body to predict its category."
    )

    subject = st.text_input(
        "Email Subject"
    )

    body = st.text_area(
        "Email Body",
        height=200
    )

    if st.button("Predict Email Category"):

        if not subject.strip() and not body.strip():

            st.warning(
                "Please enter an email subject or body."
            )

        else:

            try:

                model, mlb = load_model()

                email_text = (
                    f"Subject: {subject} Body: {body}"
                )

                prediction = model.predict(
                    [email_text]
                )[0]

                if not hasattr(model, "predict_proba"):

                    st.error(
                        "This model does not support probability prediction."
                    )

                else:

                    probabilities = model.predict_proba(
                        [email_text]
                    )[0]

                    categories = mlb.classes_

                    predicted_indices = [
                        i
                        for i, value in enumerate(prediction)
                        if value == 1
                    ]

                    if predicted_indices:

                        st.success(
                            "Email classified successfully!"
                        )

                        st.subheader(
                            "📌 Predicted Categories"
                        )

                        for i in predicted_indices:

                            category = categories[i]

                            confidence = (
                                float(probabilities[i]) * 100
                            )

                            confidence = max(
                                0.0,
                                min(100.0, confidence)
                            )

                            st.markdown(
                                f"**{category}**"
                            )

                            st.progress(
                                confidence / 100.0
                            )

                            st.caption(
                                f"Model Probability: {confidence:.2f}%"
                            )

                        st.caption(
                            "These percentages are model-predicted "
                            "probabilities. They are not guaranteed "
                            "or calibrated confidence scores."
                        )

                    else:

                        st.info(
                            "No category was predicted."
                        )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# ==================================================
# 4. EMAIL ANALYTICS
# ==================================================

elif page == "Analytics":

    st.title("📊 Email Analytics")

    st.write(
        "Visual analysis of email categories in the dataset."
    )

    if "labels" in df.columns:

        all_labels = []

        for item in df["labels"].dropna():

            try:

                if isinstance(item, str):

                    parsed_labels = ast.literal_eval(item)

                else:

                    parsed_labels = item

                if isinstance(parsed_labels, list):

                    all_labels.extend(parsed_labels)

                else:

                    all_labels.append(
                        str(parsed_labels)
                    )

            except (ValueError, SyntaxError):

                all_labels.extend(
                    str(item).split(",")
                )

        labels = pd.Series(
            all_labels
        ).astype(str).str.strip()

        labels = labels[
            labels != ""
        ]

        if not labels.empty:

            counts = labels.value_counts().reset_index()

            counts.columns = [
                "Category",
                "Count"
            ]

            total_categories = len(counts)

            total_label_count = int(
                counts["Count"].sum()
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Categories",
                total_categories
            )

            col2.metric(
                "Total Category Labels",
                total_label_count
            )

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "📊 Category Distribution"
                )

                fig = px.bar(
                    counts,
                    x="Category",
                    y="Count",
                    title="Emails by Category",
                    text="Count"
                )

                fig.update_layout(
                    xaxis_title="Email Category",
                    yaxis_title="Number of Emails"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            with col2:

                st.subheader(
                    "🥧 Category Share"
                )

                pie = px.pie(
                    counts,
                    names="Category",
                    values="Count",
                    title="Category Percentage",
                    hole=0.3
                )

                st.plotly_chart(
                    pie,
                    width="stretch"
                )

            st.markdown("---")

            st.subheader(
                "Category Count Details"
            )

            st.dataframe(
                counts,
                width="stretch"
            )

        else:

            st.warning(
                "No category labels found in the dataset."
            )

    else:

        st.error(
            "The dataset does not contain a labels column."
        )


# ==================================================
# 5. MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.title(
        "📊 Machine Learning Model Evaluation"
    )

    st.write(
        "Performance comparison of four machine learning algorithms."
    )

    comparison_path = (
        BASE_DIR / "model_comparison.csv"
    )

    report_path = (
        BASE_DIR / "classification_report.txt"
    )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    if comparison_path.exists():

        results = pd.read_csv(
            comparison_path
        )

        st.subheader(
            "Model Comparison"
        )

        format_columns = {
            col: "{:.2%}"
            for col in [
                "Exact Match Accuracy",
                "Micro Precision",
                "Micro Recall",
                "Micro F1",
                "Macro F1",
                "Weighted F1",
                "Hamming Loss"
            ]
            if col in results.columns
        }

        st.dataframe(
            results.style.format(format_columns),
            width="stretch"
        )

        if (
            "Micro F1" in results.columns
            and "Model" in results.columns
        ):

            best = results.sort_values(
                "Micro F1",
                ascending=False
            ).iloc[0]

            st.success(
                f"Best Model: {best['Model']} | "
                f"Micro F1: {best['Micro F1']:.2%}"
            )

            f1_columns = [
                col
                for col in [
                    "Micro F1",
                    "Macro F1",
                    "Weighted F1"
                ]
                if col in results.columns
            ]

            if f1_columns:

                st.subheader(
                    "Model Performance Comparison"
                )

                fig = px.bar(
                    results,
                    x="Model",
                    y=f1_columns,
                    barmode="group",
                    title="F1 Score Comparison"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

    else:

        st.error(
            "Model comparison file not found."
        )


    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📋 Category-wise Classification Report"
    )

    if report_path.exists():

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as file:

            report = file.read()

        st.code(
            report,
            language="text"
        )

    else:

        st.warning(
            "Classification report not found."
        )


    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔍 Category-wise Confusion Matrix"
    )

    try:

        evaluation_df = df.copy()

        evaluation_df["subject"] = (
            evaluation_df["subject"]
            .fillna("")
            .astype(str)
        )

        evaluation_df["body"] = (
            evaluation_df["body"]
            .fillna("")
            .astype(str)
        )

        def clean_evaluation_text(text):

            text = str(text).lower()

            text = re.sub(
                r"http\S+|www\S+",
                " ",
                text
            )

            text = re.sub(
                r"\S+@\S+",
                " ",
                text
            )

            text = re.sub(
                r"[^a-z\s]",
                " ",
                text
            )

            return re.sub(
                r"\s+",
                " ",
                text
            ).strip()

        X = (
            evaluation_df["subject"]
            + " "
            + evaluation_df["body"]
        ).apply(clean_evaluation_text)

        labels = evaluation_df["labels"].apply(
            lambda item: ast.literal_eval(item)
            if isinstance(item, str)
            else item
        )

        mlb = MultiLabelBinarizer()

        y = mlb.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        best_model_path = (
            BASE_DIR
            / "Models"
            / "email_classifier_best.pkl"
        )

        evaluation_model = joblib.load(
            best_model_path
        )

        predictions = evaluation_model.predict(
            X_test
        )

        matrices = multilabel_confusion_matrix(
            y_test,
            predictions
        )

        category = st.selectbox(
            "Select Email Category",
            mlb.classes_
        )

        index = list(
            mlb.classes_
        ).index(category)

        matrix = matrices[index]

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Actual: No",
                "Actual: Yes"
            ],
            columns=[
                "Predicted: No",
                "Predicted: Yes"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "True Negative",
            int(matrix[0][0])
        )

        col2.metric(
            "False Positive",
            int(matrix[0][1])
        )

        col3.metric(
            "False Negative",
            int(matrix[1][0])
        )

        col4.metric(
            "True Positive",
            int(matrix[1][1])
        )

        st.dataframe(
            matrix_df,
            width="stretch"
        )

        fig_cm = px.imshow(
            matrix,
            text_auto=True,
            x=[
                "Predicted: No",
                "Predicted: Yes"
            ],
            y=[
                "Actual: No",
                "Actual: Yes"
            ],
            title=f"Confusion Matrix - {category}",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            fig_cm,
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Could not generate confusion matrix: {e}"
        )