import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATASET
train_df = pd.read_csv(
    "train_data.txt",
    sep=":::",
    names=["id", "genre", "plot"],
    engine="python"
)

# 2. CLEAN LABELS
train_df["genre"] = train_df["genre"].astype(str).str.strip().str.lower()

# 3. PREPARE DATA
X = train_df["plot"].astype(str)
y = train_df["genre"]

# 4. TRAIN–VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. TF-IDF VECTORIZATION
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# 6. TRAIN MODEL
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. PREDICT & EVALUATE

y_pred = model.predict(X_val_tfidf)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))
