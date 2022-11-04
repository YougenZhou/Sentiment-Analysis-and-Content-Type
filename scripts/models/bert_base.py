import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        if args.task == 'SentimentClassifier':
            num_cls = 3
        else:
            num_cls = 5
        self.encoder = BertModel.from_pretrained('./package/bert_base_chinese')
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = nn.LayerNorm(768)
        self.dense_ac = nn.GELU()
        self.cls = nn.Linear(768, num_cls)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, token_ids, labels):
        outputs = self.encoder(token_ids)['pooler_output']
        outputs = self.cls(self.dense_ac(self.LayerNorm(self.dense(outputs))))
        loss = self.loss_fn(outputs, labels.squeeze(1))
        return outputs, loss
