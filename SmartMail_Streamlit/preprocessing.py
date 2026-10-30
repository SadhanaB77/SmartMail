
import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset
df = pd.read_csv("emails_dataset.csv")

# --------------------------------
# 1. Handle Missing Values
# --------------------------------

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

# --------------------------------
# 2. Combine Subject and Body
# --------------------------------

df["email_text"] = (
    df["subject"].astype(str) + " " +
    df["body"].astype(str)
)

# --------------------------------
# 3. Text Cleaning
# --------------------------------

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_text"] = df["email_text"].apply(clean_text)

# --------------------------------
# 4. Convert Labels into Lists
# --------------------------------

df["labels"] = df["labels"].apply(ast.literal_eval)

# --------------------------------
# 5. Convert Labels into Binary Format
# --------------------------------

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(df["labels"])

# --------------------------------
# 6. TF-IDF Feature Extraction
# --------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df["cleaned_text"])

# --------------------------------
# 7. Display Results
# --------------------------------

print("\n========== PREPROCESSING RESULTS ==========")

print("Total Emails:", len(df))

print("Original Text Example:")
print(df["email_text"].iloc[0][:200])

print("\nCleaned Text Example:")
print(df["cleaned_text"].iloc[0][:200])

print("\nTF-IDF Matrix Shape:", X.shape)

print("Label Matrix Shape:", y.shape)

print("\nCategories:")
print(mlb.classes_.tolist())

print("\n========== PREPROCESSING COMPLETED ==========")
