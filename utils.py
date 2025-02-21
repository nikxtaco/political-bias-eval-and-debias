import torch
import torch.nn as nn
from transformers import AutoModel

# Load the pre-trained embedding model (no classification head)
class BiasClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super(BiasClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_path)  # Embedding model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)  # Classification layer

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # Use CLS token embedding
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        
        return logits