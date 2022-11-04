import argparse
import os
import pickle
import random
import warnings

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

from scripts.utils import parse_args, str2bool
from scripts import models

warnings.filterwarnings('ignore')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='content',
                        choices=['sentiment', 'content'])
    parser.add_argument('--model', type=str, default='Both',
                        choices=['Bert', 'BiLSTM', 'Both'])
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_metric', type=str, default='accuracy')
    parser.add_argument('--is_application', type=str2bool, default=False)

    args = parse_args(parser)
    args.use_cuda = torch.cuda.is_available()
    return args


def get_pretrained_embeddings(task):
    if task == 'sentiment':
        excel_file = './data/Data2_Sentiment polarity.xlsx'
    elif task == 'content':
        excel_file = './data/Data1_Content type.xlsx'
    else:
        raise ValueError(f'Unsupported Task: {task}')

    data_frame = read_data_from_excel(excel_file, sheet_name=0, usecols=[0], names=['comment'])
    vocab_path = f'./package/lstm_base_chinese/{task}_vocab.pkl'
    pretrained_embeddings_path = './package/lstm_base_chinese/sgns.sogou.char'

    embeddings_size = 300
    filename_base = f'./package/lstm_base_chinese/embedding_{task}'
    if os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, 'rb'))
    else:
        vocab = build_vocab(data_frame, task)

    if not os.path.exists(f'./package/lstm_base_chinese/embedding_{task}.npz'):
        embeddings = np.random.rand(len(vocab), embeddings_size)
        f = open(pretrained_embeddings_path, 'r', encoding='utf-8')
        for i, line in enumerate(f.readlines()):
            lin = line.strip().split(' ')
            if lin[0] in vocab:
                idx = vocab[lin[0]]
                emb = [float(x) for x in lin[1: 301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        f.close()
        np.savez_compressed(filename_base, embeddings=embeddings)

    pretrained_embedding = np.load(f'./package/lstm_base_chinese/embedding_{task}.npz')['embeddings'].astype('float32')
    pretrained_embedding = torch.tensor(pretrained_embedding)
    return pretrained_embedding


def build_vocab(data_frame, task, max_size=10000, min_freq=1):
    vocab = {}
    tokenizer = lambda x: [y for y in x]
    for comment in data_frame.comment:
        line = comment.strip()
        if not line:
            continue
        for char in tokenizer(line):
            vocab[char] = vocab.get(char, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab = {char_count[0]: idx for idx, char_count in enumerate(vocab_list)}
    vocab.update({'<UNK>': len(vocab), '<PAD>': len(vocab) + 1})
    pickle.dump(vocab, open(f'./package/lstm_base_chinese/{task}_vocab.pkl', 'wb'))
    return vocab


def read_data_from_excel(excel_file, sheet_name=0, usecols=None, names=None):
    data_frame = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=usecols, names=names)
    data_frame.dropna(inplace=True)
    return data_frame


def gen_data(task, k_fold):
    if task == 'sentiment':
        excel_file = './data/Data2_Sentiment polarity.xlsx'
        cls_name = 'polarity'
    elif task == 'content':
        excel_file = './data/Data1_Content type.xlsx'
        cls_name = 'ctype'
    else:
        raise ValueError(f'Unsupported Task: {task}')
    data_frame = read_data_from_excel(excel_file, sheet_name=0, usecols=[0, 1], names=['comment', cls_name])
    k_fold_split(data_frame, k_fold, task)


def k_fold_split(data_frame, k, task):
    k_fold = []
    index = set(range(data_frame.shape[0]))
    for i in range(k):
        if i == k - 1:
            k_fold.append(list(index))
        else:
            tmp = random.sample(list(index), int(1.0 / k * data_frame.shape[0]))
            k_fold.append(tmp)
            index -= set(tmp)

    for i in range(k):
        tra = []
        dev = k_fold[i]
        for j in range(k):
            if i != j:
                tra += k_fold[j]
        data_frame.iloc[tra].to_csv(f'./data/{k}_fold/{task}_train_{i}', sep=',', index=False)
        data_frame.iloc[dev].to_csv(f'./data/{k}_fold/{task}_valid_{i}', sep=',', index=False)
    print('数据拆分完成！')


def train(args, model, optimizer, k):
    best_metric = 0.0
    best_result = {}
    patient = 0
    eval_metric = args.eval_metric

    train_data = load_k_fold_data(args.task, args.k_fold, k, tag='train')
    train_set = TextDataset(args, train_data)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)

    for epoch in range(args.num_epochs):
        metrics = None
        print('=' * 80)
        for step, (x, y) in enumerate(train_loader):
            model.train()
            x, y = gen_inputs(x, y)
            logit, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            part_metrics = gen_metrics(logit, y)
            metrics = merge_part_metrics(part_metrics, metrics)
        metrics = average_metrics(metrics, len(train_loader))
        print(f'[Train: {epoch + 1}/{args.num_epochs}]' + ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))

        test_metrics = evaluation(args, model, k)
        print(f'[Evaluation: {epoch + 1}]' + ', '.join(f'{k}: {v:.4f}' for k, v in test_metrics.items()))

        if test_metrics[eval_metric] > best_metric:
            patient = 0
            best_metric = test_metrics[eval_metric]
            best_result = test_metrics
            torch.save(model.state_dict(), f'./output/{args.task}_{args.model}_best.pth')
        else:
            patient += 1

        if patient > 10:
            return best_result

    return best_result


def evaluation(args, model, k):
    data = load_k_fold_data(args.task, args.k_fold, k, tag='valid')
    dev_set = TextDataset(args, data)
    loader = DataLoader(dev_set, args.batch_size, shuffle=False, collate_fn=dev_set.collate_fn)
    metrics = None
    model.eval()
    for step, (x, y) in enumerate(loader):
        with torch.no_grad():
            x, y = gen_inputs(x, y)
            outputs, loss = model(x, y)
            part_metrics = gen_metrics(outputs, y, tag='valid')
        metrics = merge_part_metrics(part_metrics, metrics)
    metrics = average_metrics(metrics, len(loader))
    return metrics


def average_metrics(metrics, steps):
    outputs = {}
    for k in metrics:
        outputs[k] = metrics[k] / steps
    return outputs


def merge_part_metrics(part_metrics, metrics):
    if metrics is None:
        return part_metrics
    if part_metrics is None:
        return metrics
    new_metrics = {}
    for k in metrics:
        new_metrics[k] = metrics[k] + part_metrics[k]
    return new_metrics


def gen_metrics(outputs, labels, tag='train'):
    metrics = {}
    logits = torch.max(outputs, 1)[1].data.cpu()
    labels = labels.squeeze(-1).data.cpu()
    metrics['precision'] = precision_score(labels, logits, average='macro')
    metrics['recall'] = recall_score(labels, logits, average='macro')
    metrics['accuracy'] = accuracy_score(labels, logits)
    metrics['f1_measure'] = f1_score(labels, logits, average='macro')
    if tag == 'valid':
        metrics['kappa_score'] = cohen_kappa_score(logits, labels)
        metrics['pearson_score'] = pearsonr(logits, labels)[0]
        metrics['spearman_score'] = spearmanr(logits, labels)[0]
    return metrics


def gen_inputs(x, y):
    return torch.tensor(x).cuda(), torch.tensor(y).cuda()


def load_k_fold_data(task, k_fold, k, tag):
    data_frame = pd.read_csv(f'./data/{k_fold}_fold/{task}_{tag}_{k}')
    data_frame.index = range(len(data_frame))
    return data_frame


class TextDataset(Dataset):
    def __init__(self, args, data_frame):
        if args.task == 'content':
            self.cls_name = 'ctype'
            self.label_map = {'C': 0, 'R': 1, 'S': 2, 'D': 3, 'Q': 4, 'R S': 1, 'R C': 1, 'C S': 0, 'C R': 0, 'IR': 1}
        else:
            self.cls_name = 'polarity'
            self.label_map = {'1': 0, '0': 1, '-1': 2, '-2': 2, '2': 1, '9': 0, '3': 1,
                              'O': 1, 'R': 0, 'S': 1, '1.0': 0, '-1.0': 2, '0.0': 1}
        if args.model.lower() == 'bilstm':
            self.tokenizer = lambda x: [y for y in x]
            self.vocab = pickle.load(open(f'./package/lstm_base_chinese/{args.task}_vocab.pkl', 'rb'))
            self.pad_id = self.vocab.get('<PAD>', 0)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('./package/bert_base_chinese')
            self.pad_id = 0
        self.data_frame = data_frame
        self.model = args.model
        self.task = args.task
        self.is_application = args.is_application

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        comment = self.data_frame.comment[index]
        label = self.data_frame[self.cls_name][index]
        if self.model.lower() == 'bilstm':
            comment_tokens = self.tokenizer(comment)
            if len(comment_tokens) < 32:
                comment_tokens.extend(['<PAD>'] * (32 - len(comment_tokens)))
            else:
                comment_tokens = comment_tokens[: 32]
            comment_ids = [self.vocab.get(token, self.vocab.get('<UNK>')) for token in comment_tokens]
        else:
            comment_tokens = self.tokenizer.tokenize(comment)
            comment_ids = self.tokenizer.convert_tokens_to_ids(comment_tokens)
        label_id = self.label_map[str(label).strip()]
        if self.is_application:
            return comment_ids, label_id, index
        else:
            return comment_ids, label_id

    def collate_fn(self, examples):
        batch_token_list = [example[0] for example in examples]
        batch_label_list = [example[1] for example in examples]
        batch_token_ids = self.pad_batch_list(batch_token_list, pad_id=self.pad_id)
        batch_label_ids = np.array(batch_label_list).reshape([-1, 1])
        if self.is_application:
            batch_index = [example[2] for example in examples]
            return batch_token_ids, batch_label_ids, batch_index
        else:
            return batch_token_ids, batch_label_ids

    def pad_batch_list(self, lis, pad_id=0):
        max_len = max(map(len, lis))
        inst = np.array([list(li) + [pad_id] * (max_len - len(li)) for li in lis])
        return inst.astype('int64').reshape([-1, max_len])


def gen_logit_by_model(args):
    excel_file = './data/application.xlsx'
    model_name = 'Bert'

    type_map = {'0': 'C', '1': 'R', '2': 'S', '3': 'D', '4': 'Q'}
    polarity_map = {'0': '1', '1': '0', '2': '-1'}

    data_frame_1 = read_data_from_excel(excel_file, sheet_name=0, usecols=[2, 3, 4],
                                        names=['comment', 'type', 'polarity'])
    data_frame_2 = read_data_from_excel(excel_file, sheet_name=1, usecols=[2, 3, 4],
                                        names=['comment', 'type', 'polarity'])
    data_frame = pd.concat([data_frame_1, data_frame_2])
    data_frame.index = range(len(data_frame))

    dataset = TextDataset(args, data_frame)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    sentiment_model_cls = getattr(models, args.model)
    sentiment_model = sentiment_model_cls(args)
    sentiment_state_dict = torch.load(f'./output/sentiment_{model_name}_best.pth', map_location='cpu')
    sentiment_model.load_state_dict(sentiment_state_dict)
    sentiment_model.cuda()

    type_model_cls = getattr(models, args.model)
    type_model = type_model_cls(args)
    type_state_dict = torch.load(f'./output/content_{model_name}_best.pth', map_location='cpu')
    type_model.load_state_dict(type_state_dict)
    type_model.cuda()

    new_df = pd.DataFrame(columns=['comment', 'type_by_model', 'type', 'sentiment_by_model', 'polarity'])
    for step, (x, y, index_list) in enumerate(data_loader):
        sentiment_model.eval()
        type_model.eval()
        with torch.no_grad():
            x, y = gen_inputs(x, y)
            sentiment_logit, _ = sentiment_model(x, y)
            type_logit, _ = type_model(x, y)
            sentiment_pred = torch.max(sentiment_logit, 1)[1].data.cpu().numpy()
            type_pred = torch.max(type_logit, 1)[1].data.cpu().numpy()
            for i, idx in enumerate(index_list):
                new_df.loc[idx] = {
                    'comment': data_frame.comment[idx],
                    'type_by_model': type_map[str(type_pred[i])],
                    'type': data_frame.type[idx],
                    'sentiment_by_model': polarity_map[str(sentiment_pred[i])],
                    'polarity': data_frame.polarity[idx]
                }
    new_df.to_excel('./results/test.xlsx')
    print('标注结束！')


if __name__ == '__main__':

    args = setup_args()
    args.display()

    if args.is_application:
        gen_logit_by_model(args)
    else:
        gen_data(args.task, args.k_fold)
        result = None
        for i in range(args.k_fold):
            print(f'新的一轮交叉验证：{i + 1}')

            model_cls = getattr(models, args.model)
            if args.model.lower() == 'bilstm':
                pretrained_embeddings = get_pretrained_embeddings(args.task)
                model = model_cls(args, pretrained_embeddings)
            else:
                model = model_cls(args)
            if args.use_cuda:
                model.cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            part_result = train(args, model, optimizer, i)
            print(f'[{i + 1}_Fold_Evaluation Result: ]' + ', '.join(f'{k}: {v:.4f}' for k, v in part_result.items()))
            result = merge_part_metrics(part_result, result)

        result = average_metrics(result, args.k_fold)
        print('>' * 80)
        print(f'[K_Fold_Evaluation Result: ]' + ', '.join(f'{k}: {v:.4f}' for k, v in result.items()))
