import torch.nn as nn
from transformers import DistilBertModel, BertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', freeze=True):
        super().__init__()
        if 'bert' in model_name and 'distil' not in model_name:
            self.model = BertModel.from_pretrained(model_name)
            hidden_dim = 768
        else:
            self.model = DistilBertModel.from_pretrained(model_name)
            hidden_dim = 768

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # CLS token
