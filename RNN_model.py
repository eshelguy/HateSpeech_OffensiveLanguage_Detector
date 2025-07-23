# improved_rnn_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from clean_dataset import load_and_clean_dataset, split_data

# --- Hyperparameters ---
MAX_LEN = 50
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10  # Increased epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dataset Class ---
class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Improved RNN Model ---
class ImprovedRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(ImprovedRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Use LSTM instead of vanilla RNN for better gradient flow
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)

        # Account for bidirectional LSTM (hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding + dropout
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = self.dropout1(x)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim*2]

        # Use mean pooling instead of just final state
        x = torch.mean(lstm_out, dim=1)  # [batch, hidden_dim*2]

        # Fully connected layers with dropout
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))  # [batch, hidden_dim]
        x = self.fc2(x)  # [batch, output_dim]

        return x


# --- Tokenization, Padding, and Label Encoding ---
def preprocess_data(X_train, X_test, y_train, y_test):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    return X_train_pad, X_test_pad, y_train_enc, y_test_enc, label_encoder


# --- Create Data Loaders with Class Balancing ---
def create_loaders(X_train_pad, X_test_pad, y_train_enc, y_test_enc):
    # Calculate class weights for balanced sampling
    class_counts = np.bincount(y_train_enc)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train_enc]

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        TweetDataset(X_train_pad, y_train_enc),
        batch_size=BATCH_SIZE,
        sampler=sampler  # Use weighted sampler instead of shuffle
    )
    test_loader = DataLoader(TweetDataset(X_test_pad, y_test_enc), batch_size=BATCH_SIZE)
    return train_loader, test_loader


# --- Train the Model with Class Weights ---
def train_model(model, train_loader, y_train_enc):
    # Compute class weights for loss function
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train_enc),
                                         y=y_train_enc)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    return train_losses


# --- Evaluate the Model ---
def evaluate_model(model, test_loader, label_encoder):
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    # Convert classes to strings for proper display
    class_names = [str(cls) for cls in label_encoder.classes_]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                digits=3))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Improved RNN - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return y_true, y_pred


# --- Plot Training Progress ---
def plot_training_progress(train_losses):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Main Function ---
def main():
    df = load_and_clean_dataset("labeled_data.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Label distribution in training set:")
    print(y_train.value_counts())
    print("\nLabel distribution in test set:")
    print(y_test.value_counts())

    X_train_pad, X_test_pad, y_train_enc, y_test_enc, label_encoder = preprocess_data(X_train, X_test, y_train, y_test)
    train_loader, test_loader = create_loaders(X_train_pad, X_test_pad, y_train_enc, y_test_enc)

    # Create improved model
    model = ImprovedRNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=128,
        output_dim=len(np.unique(y_train_enc)),
        num_layers=2,
        dropout=0.3
    ).to(DEVICE)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {DEVICE}")

    # Train model
    train_losses = train_model(model, train_loader, y_train_enc)

    # Plot training progress
    plot_training_progress(train_losses)

    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_loader, label_encoder)


if __name__ == "__main__":
    main()