# README – Hate Speech Detection in Tweets using Classical and Deep Learning Models

## 📌 Project Overview

This project focuses on the **automatic detection of hate speech and offensive language** on Twitter. With the rise of harmful content on social media platforms, identifying and moderating toxic language has become a critical challenge in the field of Natural Language Processing (NLP). The aim is to classify tweets into three categories:

- **0** – *Hate Speech*  
- **1** – *Offensive Language*  
- **2** – *Neither*

Using a real-world dataset of tweets, we implement and compare three modeling approaches:
1. **TF-IDF + Logistic Regression** (baseline statistical model)
2. **Classic RNN** (sequence model)
3. **BERT + LoRA** (state-of-the-art transformer with parameter-efficient fine-tuning)

## 🧠 Motivation

Manual moderation of offensive content is impractical due to the massive scale of social media. During a period of global unrest and rising antisemitism, I encountered disturbing online content. This project was born out of a desire to develop automated tools that could detect, analyze, and even help counteract toxic discourse in real time.

## 📂 Files

| File | Description |
|------|-------------|
| `TF-IDF_Logistic_Regression.py` | Implements the baseline model using TF-IDF vectorization and Logistic Regression. |
| `rnn_model.py` | Implements a classic RNN model with word embeddings for text classification. |
| `bert_model.py` | Implements a BERT-based classifier with LoRA fine-tuning and data balancing techniques. |
| `clean_dataset.py` | Contains utility functions for dataset cleaning and splitting (expected to be in the same directory). |
| `labeled_data.csv` | Dataset from Kaggle – *Hate Speech and Offensive Language Detection*. (Assumed to be present) |

## 📊 Dataset

The dataset consists of ~25,000 tweets labeled via crowd-sourcing. Label distribution is highly imbalanced:

| Label | Category             | Count | Percentage |
|-------|----------------------|-------|------------|
| 0     | Hate Speech          | 1,430 | 5.8%       |
| 1     | Offensive Language   | 19,190| 77.4%      |
| 2     | Neither              | 4,163 | 16.8%      |

## 🧪 Preprocessing

- Lowercasing
- Removal of URLs, mentions, hashtags, emojis, and special characters
- Keeping alphabetic characters only
- Saved both clean and original versions

Train-test split: stratified 80/20

## 🛠 Models and Methods

### 1. TF-IDF + Logistic Regression
- Uses `TfidfVectorizer` on unigrams and bigrams (max 5000 features)
- `class_weight='balanced'` to mitigate label imbalance
- Achieved the **highest overall accuracy (89.1%)** and **F1 score for Hate Speech (0.465)**

### 2. Classic RNN
- Tokenization via Keras `Tokenizer`, padded sequences
- Model: Embedding + RNN (128 hidden units) + Linear
- Trained with `CrossEntropyLoss`, Adam optimizer
- **Accuracy: 78.4%**, but lower performance on minority class

### 3. BERT + LoRA (Low-Rank Adaptation)
- Model: `bert-base-uncased` + LoRA injected in attention layers
- Class imbalance handled using `class_weight` and `WeightedRandomSampler`
- **F1 score for Hate Speech improved to 0.403**
- More robust in capturing offensive and neutral language

## 📈 Results Summary

| Model                     | Accuracy | F1 (Hate) | F1 (Offensive) | F1 (Neutral) |
|--------------------------|----------|-----------|----------------|--------------|
| TF-IDF + Logistic Reg.   | 89.1%    | 0.465     | 0.933          | 0.850        |
| RNN                      | 78.4%    | 0.356     | 0.854          | 0.754        |
| BERT + LoRA              | 84.5%    | 0.403     | 0.90           | 0.88         |

## 🔍 Key Insights

- **Simple models can compete**: TF-IDF + Logistic Regression performed surprisingly well thanks to class balancing and statistical features.
- **Context matters**: BERT + LoRA captured deeper semantic context and handled offensive/neutral categories robustly.
- **RNNs underperform without pretraining**: Lacked the representational power of transformer-based models.
- **Class imbalance** was the major challenge; techniques like `class_weight` and LoRA helped address it.

## ⚠ Limitations

- No SMOTE or data augmentation was used
- RNN did not use pretrained embeddings
- No hyperparameter optimization
- Sarcasm/irony were not explicitly handled
- Hate Speech class was too rare (5.8%)

## 🚀 Future Work

- **Apply SMOTE or back-translation** for minority class augmentation
- **Use Focal Loss or Weighted Cross-Entropy** to improve class 0 performance
- **Try lightweight transformer models** like DistilBERT for real-time applications
- **Enhance preprocessing** to preserve emoji and nuanced signals
- **Adapt to specific domains**, like antisemitic content

## 📚 References

- **Dataset**: [Kaggle - Hate Speech and Offensive Language](https://www.kaggle.com/datasets/lxqd/twitter-hate-speech)
- **Paper**: [Offensive Language Detection on Social Media using XLNet](https://arxiv.org/html/2506.21795v1)

## 👨‍💻 Author

**Guy Eshel (208846758)**  
Bachelor of Computer Science  
Project submitted for: *Advanced Models for Language Understanding*
