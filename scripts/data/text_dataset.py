from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(self, df, args):
        self.task = args.task
        self.df = df
        if args.model != 'BiLSTM':
            self.tokenizer = BertTokenizer.from_pretrained('./package/bert_base_chinese')
        if args.task == 'ContentType':
            self.type_map = {'C': 0, 'R': 1, 'S': 2, 'D': 3, 'Q': 4, 'R S': 1, 'R C': 1, 'C S': 0, 'C R': 0}
        else:
            self.polarity_map = {'1': 0, '0': 1, '-1': 2, '-2': 2, '2': 1, '9': 0, '3': 1}

    def __getitem__(self, index):
        comment = self.df.comment[index]
        comment = self.tokenizer.tokenize(comment)
        comment_ids = self.tokenizer.convert_tokens_to_ids(comment)
        if self.task == 'SentimentClassifier':
            polarity = self.df.polarity[index]
            polarity_ids = self.polarity_map[str(polarity)]
            return comment_ids, polarity_ids
        else:
            content_type = self.df.c_type[index].strip()
            try:
                content_type_id = self.type_map[str(content_type)]
            except KeyError:
                print(index, content_type)
            return comment_ids, content_type_id

    def __len__(self):
        return len(self.df)
