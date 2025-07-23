
# Hate Speech Detection on Twitter Using NLP Models

## Project Overview

This project aims to detect and classify hate speech on Twitter into three categories:
- **Hate Speech (0)**
- **Offensive Language (1)**
- **Neither (2)**

We explored and compared three modeling approaches:
1. **TF-IDF + Logistic Regression** (baseline statistical model)
2. **Classic RNN** (sequence model)
3. **BERT + LoRA** (state-of-the-art transformer with parameter-efficient fine-tuning)

## Motivation

Manual moderation of offensive content is impractical due to the massive scale of social media. During a period of global unrest and rising antisemitism, I encountered disturbing online content. This project was born out of a desire to develop automated tools that could detect, analyze, and even help counteract toxic discourse in real time.

## Repository Structure

```
├── BERT_LoRA.py              # BERT with Low-Rank Adaptation (LoRA)
├── RNN_model.py              # Improved RNN (Bi-LSTM) model
├── TF-IDF_LR.py              # Logistic Regression with TF-IDF
├── clean_dataset.py          # Preprocessing and data splitting functions
├── labeled_data.csv          # Dataset (annotated tweets)
├── הנחיות לפרויקט.pdf         # Project guidelines from course
```

## Dataset

- **Source**: Hate Speech and Offensive Language Dataset (Kaggle)
- **Size**: ~25,000 tweets
- **Imbalance**: Class 1 (Offensive) dominates (~77%), with Hate Speech (Class 0) being the minority (~6%).

## Preprocessing

- Lowercasing
- Removal of URLs, mentions, hashtags, emojis, and special characters
- Keeping alphabetic characters only
- Saved both clean and original versions

Train-test split: stratified 80/20

## Models and Methods

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

## How to Run

Install dependencies:
```
pip install torch transformers scikit-learn matplotlib seaborn peft
```

Run each model independently:
```
python TF-IDF_LR.py
python RNN_model.py
python BERT_LoRA.py
```

## Results Summary

| Model                    | Accuracy | F1 (Hate Speech) | F1 (Offensive) |
|-------------------------|----------|------------------|----------------|
| TF-IDF + Logistic Reg.  | 89.1%    | 0.465            | 0.933          |
| RNN (Bi-LSTM)           | 78.4%    | 0.356            | 0.854          |
| BERT + LoRA             | 84.5%    | 0.403            | 0.90+          |

## Key Insights

- Data imbalance heavily affects Hate Speech detection.
- TF-IDF + LR outperformed expectations.
- LoRA improves BERT’s efficiency without sacrificing too much performance.

## References

- **Dataset**: [Kaggle - Hate Speech and Offensive Language](https://www.kaggle.com/datasets/lxqd/twitter-hate-speech)
- **Paper**: [Offensive Language Detection on Social Media using XLNet](https://arxiv.org/html/2506.21795v1)
