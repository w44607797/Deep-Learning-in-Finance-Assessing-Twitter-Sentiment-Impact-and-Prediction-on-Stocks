import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
# Function to evaluate the model
# Function to evaluate the model
# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()  # Evaluation mode
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move the batch to the same device as the model
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            # Perform the forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            # Update counts
            true_positives += (predictions & labels).sum().item()
            true_negatives += ((~predictions) & (~labels)).sum().item()
            false_positives += (predictions & (~labels)).sum().item()
            false_negatives += ((~predictions) & labels).sum().item()

    # Calculate evaluation metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('./results/checkpoint-500')  # Update this path


file_path = './stock_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Preprocess the Sentiment labels: convert -1 to 0 and 1 to 1
data['Sentiment'] = data['Sentiment'].map({-1: 0, 1: 2})

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['Text'], data['Sentiment'],
    test_size=0.1,  # 10% for validation
    random_state=42
)


# Tokenize the validation texts
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

# Create a TensorDataset from the tokenized texts
val_dataset = TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_labels.tolist())
)

# Create a DataLoader for the validation set
val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=64  # Adjust batch size according to your needs
)

# Load the old and new models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
old_model = AutoModelForSequenceClassification.from_pretrained('./models').to(device)


# Evaluate the old model
old_precision, old_recall, old_f1 = evaluate_model(old_model, val_dataloader,device)
print(f"without pretrain Model - Precision: {old_precision}, Recall: {old_recall}, F1: {old_f1}")


checkpoint = [500,1000,1500,2000,2500,3000]

# Evaluate the new model
for i in checkpoint:
    new_model = AutoModelForSequenceClassification.from_pretrained(f'./results_10/checkpoint-{i}').to(device)
    new_precision, new_recall, new_f1 = evaluate_model(new_model, val_dataloader,device)
    print(f"checkpoint{i} Model -  Precision: {new_precision}, Recall: {new_recall}, F1: {new_f1}")

