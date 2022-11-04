import argparse
import logging
import os
import warnings

import torch.cuda
from torch.utils.data import DataLoader

from scripts.utils import parse_args, read_data, pad_batch_data, build_dataset, build_iterator, load_embeddings
from scripts import models, tasks
from scripts.data.text_dataset import TextDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
_logger = logging.getLogger('main')

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
    args.display()
    return args


def main(args):
    model_cls = getattr(models, args.model)
    if args.model != 'BiLSTM':
        train_df, valid_df, test_df = read_data(args.task)
        train_set = TextDataset(train_df, args)
        valid_set = TextDataset(valid_df, args)
        test_set = TextDataset(test_df, args)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch_data)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=pad_batch_data)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=pad_batch_data)
        model = model_cls(args)
    else:
        vocab, train_data, dev_data, test_data = build_dataset(args, False)
        train_loader = build_iterator(train_data, args, shuffle=True)
        valid_loader = build_iterator(dev_data, args)
        test_loader = build_iterator(test_data, args)
        pretrained_embeddings = load_embeddings(args)
        model = model_cls(pretrained_embeddings)
    if args.use_cuda:
        model = model.cuda()

    task_cls = getattr(tasks, args.task)
    task = task_cls(args, model)

    best_metric = 0.0
    patient = 0

    for epoch in range(args.num_epochs):
        for step, (token_ids, polarity) in enumerate(train_loader):
            model.train()
            train_metrics = task.train_step(model, token_ids, polarity)
            if (step + 1) % args.log_steps == 0 or (step + 1) == len(train_loader):
                print(f'[Train: {epoch + 1}/{args.num_epochs}-{step + 1}]' +
                      ', '.join(f'{k}: {v:.4f}' for k, v in train_metrics.items()))

        print('=' * 80)
        valid_metrics = task.evaluation(model, valid_loader)
        print(f'[Evaluation: {epoch + 1}]' + ', '.join(f'{k}: {v:.4f}' for k, v in valid_metrics.items()))

        test_metrics = task.evaluation(model, test_loader)
        print(f'[Evaluation: {epoch + 1}]' + ', '.join(f'{k}: {v:.4f}' for k, v in test_metrics.items()))
        print('=' * 80)

        if test_metrics[args.eval_metric] >= best_metric:
            best_metric = test_metrics[args.eval_metric]
            best_result = test_metrics
            patient = 0
            torch.save(model.state_dict(), f'./output/{args.task}_{args.model}_best.pth')
        else:
            patient += 1

        if patient >= 10:
            print(', '.join(f'{k}: {v:.4f}' for k, v in best_result.items()))
            break


if __name__ == '__main__':
    args = setup_args()
    main(args)
