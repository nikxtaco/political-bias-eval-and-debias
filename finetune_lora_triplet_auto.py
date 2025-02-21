import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AdamW
import pandas as pd
from torch.utils.data import Dataset
from safetensors.torch import load_file
from collections import OrderedDict

# Configuration
MODEL_PATH = "models/models_w_lora/triplet_lora_2_epochs"  # Path to your model
NUM_LABELS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_PATH = "processed_data/random/train.pkl"
VALID_DATA_PATH = "processed_data/random/valid.pkl"
SEED = 42

# Load dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        df = pd.read_pickle(file_path)
        self.texts = df['content'].tolist()
        self.labels = df['bias'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Create datasets
train_dataset = TextDataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN)
valid_dataset = TextDataset(VALID_DATA_PATH, tokenizer, MAX_LEN)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)

# Load safe_tensors file and rename keys
adapter_state_dict = load_file("models/triplet_lora_2_epochs/adapter_model.safetensors")

# Rename keys in adapter_state_dict to match the model's expected format
modified_adapter_state_dict = OrderedDict()
for key, value in adapter_state_dict.items():
    new_key = key
    if "encoder.layer" in key:  # Example modification, adjust as necessary
        new_key = key.replace("encoder.layer", "bert.encoder.layer")
    modified_adapter_state_dict[new_key] = value

# Load the modified adapter weights into the model
model.load_state_dict(modified_adapter_state_dict, strict=False)

# Ensure the model is in training mode and on the correct device
model.to(DEVICE).train()

# Ensure all model parameters require gradients
for param in model.parameters():
    param.requires_grad = True

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_finetune_triplet_lora_2_epochs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=SEED,
    remove_unused_columns=False  # Avoid issues with unused columns
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    optimizers=(optimizer, None)  # Ensure the optimizer is used
)

# Start training
trainer.train()
