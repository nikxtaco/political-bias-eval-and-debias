import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score, mean_absolute_error
import numpy as np
import pandas as pd
from utils import BiasClassifier

MODEL_PATH = "models/triplet_wo_lora_5_epochs"  # Load the triplet-trained embedding model
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1
BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_PATH = "processed_data/random/train.pkl"
VALID_DATA_PATH = "processed_data/random/valid.pkl"
SEED = 42
NUM_LABELS = 3  # Number of bias categories

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

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
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return { 
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Create datasets
train_dataset = TextDataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN)
valid_dataset = TextDataset(VALID_DATA_PATH, tokenizer, MAX_LEN)

# Load the pre-trained embedding model (no classification head)
# class BiasClassifier(nn.Module):
#     def __init__(self, model_path, num_labels):
#         super(BiasClassifier, self).__init__()
#         self.encoder = AutoModel.from_pretrained(model_path)  # Embedding model
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)  # Classification layer

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]   # Use CLS token embedding
#         cls_embedding = self.dropout(cls_embedding)
#         logits = self.classifier(cls_embedding)

#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(logits, labels)
#             return loss, logits
        
#         return logits

# Initialize model
model = BiasClassifier(MODEL_PATH, NUM_LABELS).to(DEVICE)
for param in model.parameters(): param.data = param.data.contiguous() # Workaround for non-contiguous tensor error

def compute_metrics(p):
    predictions = torch.tensor(p.predictions)
    predictions = torch.argmax(predictions, dim=1)

    accuracy = (predictions == torch.tensor(p.label_ids)).float().mean().item()
    macro_f1 = f1_score(p.label_ids, predictions.numpy(), average='macro')
    mae = mean_absolute_error(p.label_ids, predictions.numpy())

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'mae': mae,
    }

training_args = TrainingArguments(
    output_dir="./results_triplet_biasclassifier_wo_lora_5_epochs",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
