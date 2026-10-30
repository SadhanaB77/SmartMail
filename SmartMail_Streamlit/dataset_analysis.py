
import pandas as pd
import ast

# Load dataset
df = pd.read_csv("emails_dataset.csv")

print("\n========== DATASET INFORMATION ==========")

print("Total Emails:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumn Names:")
print(df.columns.tolist())

# Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Duplicate emails
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate Rows:", df.duplicated().sum())

# Extract categories
all_labels = []

for item in df["labels"].dropna():
    labels = ast.literal_eval(item)
    all_labels.extend(labels)

# Category distribution
print("\n========== CATEGORY DISTRIBUTION ==========")

category_counts = pd.Series(all_labels).value_counts()

print(category_counts)

print("\nTotal Unique Categories:", len(category_counts))

# Subject and body statistics
print("\n========== TEXT INFORMATION ==========")

print("Empty Subjects:", df["subject"].isna().sum())
print("Empty Bodies:", df["body"].isna().sum())

print("\n========== ANALYSIS COMPLETED ==========")
