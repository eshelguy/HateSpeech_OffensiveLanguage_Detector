import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Step 1: Cleaning Function ---
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"#\w+", "", text)  # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text


# --- Load, Clean, and Return Dataset ---
def load_and_clean_dataset(path="labeled_data.csv"):
    df = pd.read_csv(path)

    if "tweet" not in df.columns:
        raise ValueError("Dataset must contain a 'tweet' column.")

    df["clean_tweet"] = df["tweet"].astype(str).apply(clean_tweet)

    return df



def split_data(df, text_column="clean_tweet", label_column="class", test_size=0.2, random_state=42):
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError("DataFrame must contain the specified text and label columns.")

    X = df[text_column]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Show label distribution in both sets
    print("Label distribution in training set:")
    print(y_train.value_counts().sort_index().rename(lambda x: f"Class {x}"))

    print("\nLabel distribution in test set:")
    print(y_test.value_counts().sort_index().rename(lambda x: f"Class {x}"))

    return X_train, X_test, y_train, y_test


def balance_dataset(df):
    # Separate classes
    class_0 = df[df['class'] == 0]  # Hate Speech
    class_1 = df[df['class'] == 1].sample(n=3500, random_state=42)  # Downsample
    class_2 = df[df['class'] == 2]  # Neutral

    # Upsample class_0 x3
    balanced_df = pd.concat([class_0] * 3 + [class_1, class_2], axis=0).sample(frac=1, random_state=42)

    # Optional: plot the class distribution
    plt.pie(balanced_df['class'].value_counts().values,
            labels=balanced_df['class'].value_counts().index,
            autopct='%1.1f%%')
    plt.title("Balanced Class Distribution")
    plt.show()

    return balanced_df