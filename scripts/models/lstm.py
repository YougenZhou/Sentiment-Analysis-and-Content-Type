import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, args, pretrained_embeddings):
        super(BiLSTM, self).__init__()
        num_cls = 5 if args.task == 'content' else 3
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(300, 128, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(128 * 2, num_cls)
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        text = x
        embedded = self.dropout(self.embedding(text))
        hidden, outputs = self.rnn(embedded)
        logits = self.fc(hidden[:, -1, :])
        loss = self.loss_fn(logits, labels.squeeze(-1))
        return logits, loss
