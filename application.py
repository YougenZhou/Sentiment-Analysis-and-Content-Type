import os
import pickle as pkl
import warnings

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from scripts import models
from scripts.utils import *

warnings.filterwarnings('ignore')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ContentType',
                        choices=['SentimentClassifier', 'ContentType'])
    parser.add_argument('--model', type=str, default='Both',
                        choices=['Bert', 'BiLSTM', 'Both'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--log_steps', type=int, default=40)
    parser.add_argument('--eval_metric', type=str, default='accuracy')

    args = parse_args(parser)
    args.use_cuda = torch.cuda.is_available()
    return args


def read_df(path):
    df_1 = pd.read_excel(path, sheet_name=0, usecols=[2, 3, 4], names=['comment', 'c_type', 'polarity'])
    df_2 = pd.read_excel(path, sheet_name=1, usecols=[2, 3, 4], names=['comment', 'c_type', 'polarity'])
    df_1.dropna(inplace=True)
    df_2.dropna(inplace=True)
    df = pd.concat([df_1, df_2])
    df.index = range(len(df))
    return df


def gen_df(df):
    random_df = df.sample(frac=0.5)
    random_df.index = range(len(random_df))
    return random_df


class CommentDataset(Dataset):
    def __init__(self, df, task='sentiment', model='bert'):
        super(CommentDataset, self).__init__()
        self.df = df
        self.task = task
        self.model = model
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        if model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('./package/bert_base_chinese')
            self.pad_id = 0
        else:
            self.tokenizer = lambda x: [y for y in x]
            if os.path.exists(f'./package/lstm_base_chinese/{self.task}_vocab.pkl'):
                self.vocab = pkl.load(open(f'./package/lstm_base_chinese/{self.task}_vocab.pkl', 'rb'))
            else:
                self.vocab = self._build_vocab(max_size=10000, min_freq=1)
            self.pad_id = self.vocab.get(self.pad)
        if self.task == 'content':
            self.type_map = {'C': 0, 'R': 1, 'S': 2, 'D': 3, 'Q': 4, 'R S': 1, 'R C': 1, 'C S': 0, 'C R': 0, 'IR': 1}
        else:
            self.polarity_map = {'1': 0, '0': 1, '-1': 2, '-2': 2, '2': 1, '9': 0, '3': 1, 'O': 1, 'R': 0, 'S': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        comment = self.df.comment[index]
        if self.task == 'sentiment':
            label = self.df.polarity[index]
            label_id = self.polarity_map[str(label)]
        elif self.task == 'content':
            label = self.df.c_type[index].strip()
            label_id = self.type_map[str(label)]
        if self.model == 'bert':
            comment_tokens = self.tokenizer.tokenize(comment)
            comment_ids = self.tokenizer.convert_tokens_to_ids(comment_tokens)
        else:
            comment_tokens = self.tokenizer(comment)
            comment_ids = [self.vocab.get(token, self.vocab.get(self.unk)) for token in comment_tokens]

        return comment_ids, label_id

    def _build_vocab(self, max_size=10000, min_freq=1):
        vocab_dic = {}
        for comment in self.df.comment:
            lin = comment.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in self.tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({self.unk: len(vocab_dic), self.pad: len(vocab_dic) + 1})
        pkl.dump(vocab_dic, open(f'./package/lstm_base_chinese/{self.task}_vocab.pkl', 'wb'))
        return vocab_dic

    def collate_fn(self, examples):
        batch_token_list = [example[0] for example in examples]
        batch_label_list = [example[1] for example in examples]
        batch_token_ids = self.pad_batch_list(batch_token_list, pad_id=self.pad_id)
        batch_label_ids = np.array(batch_label_list).reshape([-1, 1])
        return batch_token_ids, batch_label_ids

    def pad_batch_list(self, lis, pad_id=0):
        max_len = max(map(len, lis))
        inst = np.array([list(li) + [pad_id] * (max_len - len(li)) for li in lis])
        return inst.astype('int64').reshape([-1, max_len])


def gen_inputs(x, y):
    return torch.tensor(x).cuda(), torch.tensor(y).cuda()


def get_metrics(outputs, labels, loss):
    metrics = {}
    logits = torch.max(outputs, 1)[1].data.cpu()
    labels = labels.squeeze(-1).data.cpu()
    metrics['loss'] = loss.item()
    metrics['precision'] = precision_score(labels, logits, average='macro')
    metrics['recall'] = recall_score(labels, logits, average='macro')
    metrics['accuracy'] = accuracy_score(labels, logits)
    metrics['f1_measure'] = f1_score(labels, logits, average='macro')
    return metrics


def merge_part_metrics(part_metrics, metrics):
    if metrics is None:
        return part_metrics
    if part_metrics is None:
        return metrics
    new_metrics = {}
    for k in metrics:
        new_metrics[k] = metrics[k] + part_metrics[k]
    return new_metrics


def aveg_metrics(metrics, steps):
    outputs = {}
    for k in metrics:
        outputs[k] = metrics[k] / steps
    return outputs


def get_pretrained_embeddings(args):
    df = read_df('./data/application.xlsx')
    vocab_dir = f'./package/lstm_base_chinese/{args.task}_vocab.pkl'
    pretrain_dir = "./package/lstm_base_chinese/sgns.sogou.char"

    emb_dim = 300
    filename_trimmed_dir = f"./package/lstm_base_chinese/embedding_SougouNews_{args.task}"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        raise ValueError

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    embeddings_ = np.load(f'./package/lstm_base_chinese/embedding_SougouNews_{args.task}.npz')['embeddings'].astype(
        'float32')
    pretrained_embeddings = torch.tensor(embeddings_)
    return pretrained_embeddings


def main(k_f, model):
    df = read_df('./data/application.xlsx')

    # 随机生成df用于测试
    df = gen_df(df)

    dataset = CommentDataset(df, task='content', model='bert')
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    metrics = None
    for step, (x, y) in enumerate(data_loader):
        x, y = gen_inputs(x, y)
        logit, loss = model(x, y)
        part_metrics = get_metrics(logit, y, loss)
        metrics = merge_part_metrics(part_metrics, metrics)
    metrics = aveg_metrics(metrics, len(data_loader))
    print(f'[Process:{k_f + 1}] ' + ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
    return metrics


if __name__ == '__main__':

    args = setup_args()
    args.display()
    model_cls = getattr(models, args.model)
    if args.model.lower() == 'bert' or args.model.lower() == 'both':
        model = model_cls(args).cuda()
    else:
        pretrained_embeddings = get_pretrained_embeddings(args)
        model = model_cls(pretrained_embeddings).cuda()
    # state_dict = torch.load(f'./output/best.pth', map_location='cpu')
    # model.load_state_dict(state_dict)

    result = None
    for i in range(10):
        part_result = main(i, model)
        result = merge_part_metrics(part_result, result)
    result = aveg_metrics(result, 10)
    print('=' * 80)
    print(', '.join(f'{k}: {v:.4f}' for k, v in result.items()))
