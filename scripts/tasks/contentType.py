import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


class ContentType(object):
    def __init__(self, args, model):
        self.use_cuda = args.use_cuda
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def _gen_inputs(self, data):
        return torch.tensor(data).cuda() if self.use_cuda else torch.tensor(data)

    def _gen_metrics(self, labels, outputs, loss):
        metrics = {}
        logits = torch.max(outputs, 1)[1].data.cpu()
        labels = labels.squeeze(-1).data.cpu()
        metrics['loss'] = loss.item()
        metrics['precision'] = precision_score(labels, logits, average='macro')
        metrics['recall'] = recall_score(labels, logits, average='macro')
        metrics['accuracy'] = accuracy_score(labels, logits)
        metrics['f1_measure'] = f1_score(labels, logits, average='macro')
        return metrics

    def _merge_part_metrics(self, metrics, part_metrics):
        if metrics is None:
            return part_metrics
        if part_metrics is None:
            return metrics
        new_metrics = {}
        for k in metrics:
            new_metrics[k] = metrics[k] + part_metrics[k]
        return new_metrics

    def _get_final_metrics(self, metrics, steps):
        outputs = {}
        for k in metrics:
            outputs[k] = metrics[k] / steps
        return outputs

    def train_step(self, model, data, labels):
        inputs = self._gen_inputs(data)
        labels = self._gen_inputs(labels)
        outputs, loss = model(inputs, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        metrics = self._gen_metrics(labels, outputs, loss)
        return metrics

    def evaluation(self, model, loader):
        metrics = None
        global_steps = 0
        model.eval()
        for step, (data, labels) in enumerate(loader):
            global_steps += 1
            with torch.no_grad():
                inputs = self._gen_inputs(data)
                labels = self._gen_inputs(labels)
                outputs, loss = model(inputs, labels)
                part_metrics = self._gen_metrics(labels, outputs, loss)
            metrics = self._merge_part_metrics(metrics, part_metrics)
        metrics = self._get_final_metrics(metrics, global_steps)
        return metrics
