# bert_lora_model.py

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from clean_dataset import load_and_clean_dataset, split_data
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType

# --- Hyperparameters ---
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-4  # Higher LR for LoRA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,  # Rank - controls model size vs performance tradeoff
    lora_alpha=32,  # LoRA scaling parameter
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Apply LoRA to attention layers
)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def create_balanced_dataloader(texts, labels, tokenizer, batch_size, is_training=True):
    dataset = TweetDataset(texts, labels, tokenizer)

    if is_training:
        # Create weighted sampler for class balance
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_model(model, train_loader, val_loader, y_train_enc):
    # Compute class weights for loss function
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train_enc),
                                         y=y_train_enc)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% warmup
        num_training_steps=total_steps
    )

    # Training loop
    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")

        for batch in train_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)

            # Apply class weights to loss
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_accuracy = evaluate_model_quick(model, val_loader)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        print("-" * 50)

    return train_losses, val_accuracies


def evaluate_model_quick(model, dataloader):
    """Quick evaluation for validation during training"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def evaluate_model_full(model, dataloader, label_encoder):
    """Full evaluation with classification report"""
    model.eval()
    y_true, y_pred = [], []

    eval_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in eval_bar:
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Convert classes to strings to avoid numpy integer issue
    class_names = [str(cls) for cls in label_encoder.classes_]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(cls) for cls in label_encoder.classes_]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("BERT + LoRA - Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_training_progress(train_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("Loading and preparing data...")
    df = load_and_clean_dataset("labeled_data.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Label distribution in training set:")
    print(y_train.value_counts())
    print("\nLabel distribution in test set:")
    print(y_test.value_counts())

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    num_labels = len(label_encoder.classes_)

    # Initialize tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets and dataloaders
    print("Creating dataloaders...")
    train_loader = create_balanced_dataloader(X_train, y_train_enc, tokenizer, BATCH_SIZE, is_training=True)
    test_loader = create_balanced_dataloader(X_test, y_test_enc, tokenizer, BATCH_SIZE, is_training=False)

    # Load model and apply LoRA
    print(f"\nLoading BERT model and applying LoRA...")
    print(f"Training on device: {DEVICE}")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )

    # Apply LoRA - this significantly reduces trainable parameters
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()  # Show how many parameters we're actually training
    model = model.to(DEVICE)

    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, y_train_enc)

    # Plot training progress
    plot_training_progress(train_losses, val_accuracies)

    # Final evaluation
    print("\nFinal evaluation on test set:")
    y_true, y_pred = evaluate_model_full(model, test_loader, label_encoder)
    plot_confusion_matrix(y_true, y_pred, label_encoder)

    print(f"\nFinal test accuracy: {(np.array(y_true) == np.array(y_pred)).mean():.4f}")
    print("Training completed!")


if __name__ == "__main__":
    main()