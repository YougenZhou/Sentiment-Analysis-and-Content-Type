import numpy as np
import pandas as pd
import os
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
polarity_map = {'1': 0, '0': 1, '-1': 2, '-2': 2, '2': 1, '9': 0, '3': 1}
type_map = {'C': 0, 'R': 1, 'S': 2, 'D': 3, 'Q': 4, 'R S': 1, 'R C': 1, 'C S': 0, 'C R': 0}


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    df = pd.read_excel(file_path, sheet_name=0, usecols=[0, 1], names=['comment', 'polarity'])
    df.dropna(inplace=True)
    for comment in df.comment:
        lin = comment.strip()
        if not lin:
            continue
        content = lin.split('\t')[0]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                 :max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def read_data(task):
    if task == 'SentimentClassifier':
        file_in = './data/Data2_Sentiment polarity.xlsx'
        df = pd.read_excel(file_in, sheet_name=0, usecols=[0, 1], names=['comment', 'polarity'])
        df.dropna(inplace=True)
        df.index = range(len(df))
    elif task == 'ContentType':
        file_in = './data/Data1_Content type.xlsx'
        df = pd.read_excel(file_in, sheet_name=0, usecols=[0, 1], names=['comment', 'c_type'])
        df.dropna(inplace=True)
        df.index = range(len(df))
    else:
        raise ValueError(f'Unsupported Task: {task}')
    train_length = int(len(df) * 0.7)
    dev_length = int(len(df) * 0.1) + train_length
    train_df = df[:train_length]
    train_df.index = range(len(train_df))

    valid_df = df[train_length:dev_length]
    valid_df.index = range(len(valid_df))

    test_df = df[dev_length:-1]
    test_df.index = range(len(test_df))

    return train_df, valid_df, test_df


def pad_batch_data(sentences):
    token_ids_list = []
    polarity_ids_list = []
    for data in sentences:
        token_ids_list.append(data[0])
        polarity_ids_list.append(data[1])
    batch_token_ids = pad_batch_list(token_ids_list)
    batch_polarity_ids = np.array(polarity_ids_list).reshape([-1, 1])
    return batch_token_ids, batch_polarity_ids


def pad_batch_list(lis, pad_id=0):
    max_len = max(map(len, lis))
    inst = np.array([list(li) + [pad_id] * (max_len - len(li)) for li in lis])
    return inst.astype('int64').reshape([-1, max_len])


def build_dataset(args, use_word):
    args.pad_size = 32
    if args.task == 'ContentType':
        args.data_path = './data/Data1_Content type.xlsx'
    else:
        args.data_path = './data/Data2_Sentiment polarity.xlsx'

    args.vocab_path = f'./package/lstm_base_chinese/{args.task}_vocab.pkl'

    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(f'./package/lstm_base_chinese/{args.task}_vocab.pkl'):
        vocab = pkl.load(open(f'./package/lstm_base_chinese/{args.task}_vocab.pkl', 'rb'))
    else:
        vocab = build_vocab(args.data_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE,
                            min_freq=1)
        pkl.dump(vocab, open(args.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(df, pad_size=32):
        contents = []

        for idx, line in df.iterrows():
            content = line.comment
            label = line.c_type.strip() if args.task == 'ContentType' else line.polarity
            label = type_map[str(label)] if args.task == 'ContentType' else polarity_map[str(label)]
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label)))
        return contents

    train_df, valid_df, test_df = read_data(args.task)
    train = load_dataset(train_df, args.pad_size)
    dev = load_dataset(valid_df, args.pad_size)
    test = load_dataset(test_df, args.pad_size)
    return vocab, train, dev, test


class DialogueDataset(Dataset):
    def __init__(self, data):
        self.examples = data

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def build_iterator(data, args, shuffle=False):
    dataset = DialogueDataset(data)

    def collate_fn(lists):
        token_ids = [line[0] for line in lists]
        labels = [line[1] for line in lists]
        batch_token_ids = pad_batch_list(token_ids)
        batch_labels = np.array(labels).reshape([-1, 1])
        return batch_token_ids, batch_labels

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader


def load_embeddings(args):
    if args.task == 'SentimentClassifier':
        file_in = './data/Data2_Sentiment polarity.xlsx'
        df = pd.read_excel(file_in, sheet_name=0, usecols=[0, 1], names=['comment', 'polarity'])
        df.dropna(inplace=True)
        df.index = range(len(df))
    elif args.task == 'ContentType':
        file_in = './data/Data1_Content type.xlsx'
        df = pd.read_excel(file_in, sheet_name=0, usecols=[0, 1], names=['comment', 'type'])
        df.dropna(inplace=True)
        df.index = range(len(df))
    else:
        raise ValueError(f'Unsupported Task: {args.task}')

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
