
# 1. Import libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset
df = pd.read_csv("Bitext_Sample_Customer_Service_Testing_Dataset.csv")


# 3. Check the dataset
print("Dataset shape:", df.shape)
print(df.head())
print(df.isnull().sum())

# 4. Select input and target
X = df["utterance"].astype(str)
y = df["intent"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=2000))
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Test with new customer messages
messages = [
    "I want to cancel my order",
    "Where is my refund?",
    "I need to change my shipping address"
]

for message in messages:
    prediction = model.predict([message])[0]
    print("Customer:", message)
    print("Predicted intent:", prediction)
    print()