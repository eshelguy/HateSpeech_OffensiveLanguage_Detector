
# Hate Speech Detection on Twitter Using NLP Models

## 📌 Project Overview

This project aims to detect and classify hate speech on Twitter into three categories:
- **Hate Speech (0)**
- **Offensive Language (1)**
- **Neither (2)**

We explored and compared three modeling approaches:
1. **TF-IDF + Logistic Regression** – A classical, interpretable baseline.
2. **RNN (LSTM-based)** – A sequential deep learning model capturing word order.
3. **BERT + LoRA** – A state-of-the-art transformer with parameter-efficient fine-tuning.

The project was conducted as part of the course *"Advanced Models of Language Understanding"* at Ariel University.

## 📁 Repository Structure

```
├── BERT_LoRA.py              # BERT with Low-Rank Adaptation (LoRA)
├── RNN_model.py              # Improved RNN (Bi-LSTM) model
├── TF-IDF_LR.py              # Logistic Regression with TF-IDF
├── clean_dataset.py          # Preprocessing and data splitting functions
├── labeled_data.csv          # Dataset (annotated tweets)
├── Language moduls - final.pdf # Final report (in Hebrew)
├── הנחיות לפרויקט.pdf         # Project guidelines from course
```

## 📊 Dataset

- **Source**: Hate Speech and Offensive Language Dataset (Kaggle)
- **Size**: ~25,000 tweets
- **Imbalance**: Class 1 (Offensive) dominates (~77%), with Hate Speech (Class 0) being the minority (~6%).

## 🧪 Models and Methods

### 1. TF-IDF + Logistic Regression
- Vectorized tweets using 1–2 grams and stopword removal.
- Trained Logistic Regression with class balancing.
- Result: Highest accuracy (89.1%), strong performance on Class 0.

### 2. Improved RNN (Bi-LSTM)
- Used embedding + bidirectional LSTM.
- Trained with weighted loss and dropout regularization.
- Result: Moderate performance, struggled with rare class.

### 3. BERT + LoRA
- Used `bert-base-uncased` with LoRA for efficient fine-tuning.
- Applied class balancing and learning rate scheduling.
- Result: Strong context understanding, balanced performance with fewer trainable parameters.

## ⚙️ How to Run

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

## 📈 Results Summary

| Model                    | Accuracy | F1 (Hate Speech) | F1 (Offensive) |
|-------------------------|----------|------------------|----------------|
| TF-IDF + Logistic Reg.  | 89.1%    | 0.465            | 0.933          |
| RNN (Bi-LSTM)           | 78.4%    | 0.356            | 0.854          |
| BERT + LoRA             | 84.5%    | 0.403            | 0.90+          |

## 📌 Key Insights

- Data imbalance heavily affects Hate Speech detection.
- TF-IDF + LR outperformed expectations.
- LoRA improves BERT’s efficiency without sacrificing too much performance.

## 🚧 Limitations

- No advanced oversampling or data augmentation.
- RNN used no pretrained embeddings.
- Limited hyperparameter tuning.
- Sarcasm or coded language remains difficult.

## 🔭 Future Work

- Apply focal loss or advanced balancing methods.
- Add pretrained embeddings to RNN.
- Use lightweight transformers (e.g., DistilBERT).
- Try data augmentation (e.g., back-translation).

## 🧠 Author

Guy Eshel – Computer Science student at Ariel University  
Project for: *Advanced Models of Language Understanding*

