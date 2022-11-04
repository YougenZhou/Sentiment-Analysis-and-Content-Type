import torch.nn as nn
from transformers import BertModel


class Both(nn.Module):
    def __init__(self, args):
        super(Both, self).__init__()
        if args.task == 'content':
            num_class = 5
        else:
            num_class = 3
        self.encoder = BertModel.from_pretrained('./package/bert_base_chinese')
        self.rnn = nn.LSTM(768, 768, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(768 * 2, num_class)
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        embedded = self.encoder(x).last_hidden_state
        hidden, outputs = self.rnn(embedded)
        logits = self.fc(hidden[:, -1, :])
        loss = self.loss_fn(logits, y.squeeze(-1))
        return logits, loss
