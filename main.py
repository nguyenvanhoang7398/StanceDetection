import argparse
from config import Config
import os
from preprocessing import *
import utils

CONFIGS = None


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--config-path", type=str, default="config/config.json", help="Path to config file")


def load_fnc_full(config):
    fnc_loader = FncLoader(config.fnc_root)
    fnc_dataset = fnc_loader.load()
    fnc_dataset.export_cross_eval(os.path.join(config.fnc_root, "fnc_full"), config.num_folds)


def load_fnc_relation_full(config):
    fnc_relation_leader = FncRelationLoader(config.fnc_root)
    fnc_relation_dataset = fnc_relation_leader.load()
    fnc_relation_dataset.export_cross_eval(os.path.join(config.fnc_root, "fnc_relation_full"), config.num_folds)


def load_fnc_relation_split(config):
    fnc_relation_leader = FncRelationLoader(config.fnc_root)
    fnc_relation_dataset = fnc_relation_leader.load_split()
    fnc_relation_dataset.export_cross_eval(os.path.join(config.fnc_root, "fnc_relation_split"), config.num_folds)


def load_fnc_split(config):
    fnc_loader = FncLoader(config.fnc_root)
    fnc_dataset_split = fnc_loader.load_split()
    fnc_dataset_split.export_cross_eval(os.path.join(config.fnc_root, "fnc_split"), config.num_folds)


def load_re17(config):
    re17_loader = RumorEval17(config.re17_root)
    re17_dataset = re17_loader.load()
    re17_dataset.export_cross_eval(config.re17_root, config.num_folds)


def load_fnn(config):
    fnn_loader = FakeNewsNetDatasetLoader(config.fnn_root)
    fnn_dataset = fnn_loader.load(clean=False)
    fnn_dataset.export_full(os.path.join(config.fnc_root, "fnc_uncleaned.csv"))
    fnn_dataset_cleaned = fnn_loader.load(clean=True)
    fnn_dataset_cleaned.export_full(os.path.join(config.fnc_root, "fnc_cleaned.csv"))

    
def preprocess(config):
    # load_fnc_full(config)
    # load_fnc_split(config)
    # load_re17(config)
    # load_fnn(config)
    load_fnc_relation_full(config)
    load_fnc_relation_split(config)


def experiment_summary(config):
    fnc_logit_path = os.path.join(config.fnn_fnc_pred_root, "stance.logits.json")
    re17_logit_path = os.path.join(config.fnn_re17_pred_root, "stance.logits.json")
    fnc_logits = [utils.softmax(row) for row in utils.read_json(fnc_logit_path)]
    re17_logits = [utils.softmax(row) for row in utils.read_json(re17_logit_path)]

    fnc_preds_path = os.path.join(config.fnn_fnc_pred_root, "stance.tsv")
    re17_preds_path = os.path.join(config.fnn_re17_pred_root, "stance.tsv")
    test_data_path = os.path.join(config.fnn_root, "dev.tsv")

    fnc_preds = utils.read_csv(fnc_preds_path, delimiter="\t")
    re17_preds = utils.read_csv(re17_preds_path, delimiter="\t")
    test_data = utils.read_csv(test_data_path, delimiter="\t")

    assert len(fnc_preds) == len(re17_preds) == len(test_data)

    data_size = len(test_data)
    header = ["index", "source", "target", "stance",
              "fnc_predictions", "fnc_support", "fnc_deny", "fnc_comment",
              "re17_predictions", "re17_support", "re17_deny", "re17_comment"]
    summary = []
    for i in range(data_size):
        _, source, target, stance = test_data[i]
        _, fnc_pred = fnc_preds[i]
        _, re17_pred = re17_preds[i]
        fnc_support, fnc_deny, fnc_comment = fnc_logits[i]
        re17_support, re17_deny, re17_comment = re17_logits[i]
        summary.append([i, source, target, stance,
                        fnc_pred, fnc_support, fnc_deny, fnc_comment,
                        re17_pred, re17_support, re17_deny, re17_comment])

    utils.write_csv(summary, header,
                    os.path.join(config.fnn_root, "summary.csv"))


if __name__ == "__main__":
    sd_parser = argparse.ArgumentParser()
    add_arguments(sd_parser)
    CONFIGS, unparsed = sd_parser.parse_known_args()
    config_json = utils.read_json(CONFIGS.config_path)
    sd_config = Config(config_json)
    preprocess(sd_config)
    # experiment_summary(sd_config)
