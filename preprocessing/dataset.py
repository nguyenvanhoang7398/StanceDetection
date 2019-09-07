import numpy as np
import os
from sklearn.model_selection import KFold
import utils

HEADER = ["index", "source", "target", "stance"]
STS_HEADER = ["index", "genre", "filename", "year", "old_index", "source1", "source2", "sentence1",
              "sentence2", "stance"]


class BaseDataset(object):
    def __init__(self, feature_set, label_set):
        self.feature_set = feature_set
        self.label_set = label_set
        self.size = len(feature_set)
        self.validate()

    @staticmethod
    def combine(ds1, ds2):
        feature_set = ds1.feature_set + ds2.feature_set
        label_set = ds1.feature_set + ds2.feature_set
        return BaseDataset(feature_set, label_set)

    def validate(self):
        assert len(self.feature_set) == len(self.label_set)
        for i in range(self.size):
            assert len(self.feature_set[i]) == 2
            assert type(self.feature_set[i][0]) == str
            assert type(self.feature_set[i][1]) == str
            assert type(self.label_set[i]) == str
            self.validate_labels(self.label_set[i])

    @staticmethod
    def validate_labels(label):
        assert type(label) == str

    def export_cross_eval(self, output_dir, num_folds=10):
        kf = KFold(n_splits=num_folds)
        X, y = np.array(self.feature_set), np.array(self.label_set)
        fold = 1
        for train_index, test_index in kf.split(X):
            if fold >= 3:
                break
            print("Creating fold {} at {}".format(fold, output_dir))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_path = os.path.join(output_dir, "fold_{}".format(str(fold)))
            if not os.path.isdir(fold_path):
                os.mkdir(fold_path)
            self.export_full(os.path.join(fold_path, "train.tsv"), X_train, y_train, "\t")
            self.export_full(os.path.join(fold_path, "dev.tsv"), X_test, y_test, "\t")
            fold += 1

    def export_full(self, path, feature_set=None, label_set=None, delimiter=","):
        feature_set = feature_set if feature_set is not None else self.feature_set
        label_set = label_set if label_set is not None else self.label_set
        assert len(feature_set) == len(label_set)
        content = []
        for i in range(len(feature_set)):
            source, target = feature_set[i][0], feature_set[i][1]
            stance = label_set[i]
            content.append([str(i), source, target, stance])
        utils.write_csv(content, HEADER, path, delimiter)

    def export_sts_format(self, path):
        content = []
        for i in range(self.size):
            source, target = self.feature_set[i][0], self.feature_set[i][1]
            stance = self.label_set[i]
            content.append([str(i), "none", "none", "none", "none", "none", "none", source, target, stance])
        utils.write_csv(content, STS_HEADER, path, delimiter="\t")


class RelationDataset(BaseDataset):
    def __init__(self, feature_set, label_set):
        super(RelationDataset, self).__init__(feature_set, label_set)

    @staticmethod
    def validate_labels(label):
        assert label in ["related", "unrelated"]


class StanceDataset(BaseDataset):
    def __init__(self, feature_set, label_set):
        super(StanceDataset, self).__init__(feature_set, label_set)

    @staticmethod
    def validate_labels(label): 
        assert label in ["support", "deny", "comment", "unrelated"]
