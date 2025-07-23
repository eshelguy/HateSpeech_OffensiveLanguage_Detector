# tfidf_logistic_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from clean_dataset import load_and_clean_dataset, split_data


# --- Train & Evaluate Logistic Regression with TF-IDF ---
def run_tfidf_logistic_regression(X_train, X_test, y_train, y_test):
    print("Label distribution in training set:")
    print(y_train.value_counts())
    print("\nLabel distribution in test set:")
    print(y_test.value_counts())

    # TF-IDF Vectorization
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        stop_words='english'  # Remove common English stop words
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Logistic Regression Model with balanced class weights
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        solver='liblinear'  # Good for small datasets with L1/L2 regularization
    )

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    # Get actual class names from the data
    unique_classes = sorted(y_train.unique())
    class_names = [str(cls) for cls in unique_classes]

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names,
                                digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("TF-IDF + Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return model, vectorizer, y_pred


# --- Main Function ---
def main():
    # Load and prepare data
    print("Loading and cleaning dataset...")
    df = load_and_clean_dataset("labeled_data.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Run TF-IDF + Logistic Regression
    model, vectorizer, y_pred = run_tfidf_logistic_regression(X_train, X_test, y_train, y_test)

    print(f"\nModel training completed!")
    print(f"Test accuracy: {(y_test == y_pred).mean():.3f}")


if __name__ == "__main__":
    main()
