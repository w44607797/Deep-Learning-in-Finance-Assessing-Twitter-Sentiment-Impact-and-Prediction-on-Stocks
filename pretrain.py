

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# Load the dataset
file_path = './stock_data.csv'
data = pd.read_csv(file_path)

# Preprocess the Sentiment labels: convert -1 to 0 and 1 to 1
data['Sentiment'] = data['Sentiment'].map({-1: 0, 1: 2})

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['Text'], data['Sentiment'],
    test_size=0.1,  # 10% for validation
    random_state=42
)

# Load the tokenizer for the pre-trained model
MODEL = "./models"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Tokenize the training and validation data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

# Convert to datasets library format
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels.tolist()})
val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'], 'labels': val_labels.tolist()})

# Load the pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define training arguments
epoch = 10
training_args = TrainingArguments(
    output_dir=f'./results_{epoch}',
    num_train_epochs=epoch,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the fine-tuned model
model.save_pretrained('path_to_save_fine_tuned_model')  # Replace with your desired save path
