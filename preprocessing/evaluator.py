from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import utils


class Evaluator(object):

    @staticmethod
    def evaluate_multi_class(y_preds, y_truths):
        pred_labels = set(y_preds)
        true_labels = set(y_truths)
        all_labels = pred_labels.union(true_labels)
        print(pred_labels, true_labels)
        label2idx, idx2label = {}, {}

        for i, label in enumerate(all_labels):
            label2idx[label] = i
            idx2label[i] = label

        preds = [label2idx[p] for p in y_preds]
        truths = [label2idx[t] for t in y_truths]

        accuracy = accuracy_score(truths, preds)
        individual_precision = precision_score(truths, preds, average=None)
        individual_recall = recall_score(truths, preds, average=None)
        individual_f1 = f1_score(truths, preds, average=None)
        micro_precision = precision_score(truths, preds, average="micro")
        micro_recall = recall_score(truths, preds, average="micro")
        micro_f1 = f1_score(truths, preds, average="micro")
        macro_precision = precision_score(truths, preds, average="macro")
        macro_recall = recall_score(truths, preds, average="macro")
        macro_f1 = f1_score(truths, preds, average="macro")
        result = {
            "accuracy": accuracy,
            "individual_precision": individual_precision,
            "individual_recall": individual_recall,
            "individual_f1": individual_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "label2idx": label2idx,
            "idx2label": idx2label
        }

        header = ["label", "accuracy", "precision", "recall", "f1"]
        csv_content = []
        table = PrettyTable(header)

        for label, idx in label2idx.items():
            row = [label, "", str(individual_precision[idx]), str(individual_recall[idx]),
                   str(individual_f1[idx])]
            table.add_row(row)
            csv_content.append(row)
        macro_row = ["macro", accuracy, macro_precision, macro_recall, macro_f1]
        micro_row = ["micro", "", micro_precision, micro_recall, micro_f1]
        table.add_row(macro_row)
        table.add_row(micro_row)
        csv_content.append(macro_row)
        csv_content.append(micro_row)
        utils.write_csv(csv_content, header, "evaluation.csv")
        print(table)
