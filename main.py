import argparse
from config import Config
from preprocessing import *
import utils

CONFIGS = None


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--config-path", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--dataset-name", type=str, default="stance_fnn", help="Dataset name to preprocess")


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


def load_re19(config):
    re19_loader = RumorEvalTwitter19(config.re19_root)
    re19_dataset = re19_loader.load()
    # re19_dataset.export_full(os.path.join(config.re19_root, "re19.csv"))
    re19_dataset.export_cross_eval(os.path.join(config.re19_root, "only_twitter"), config.num_folds)
    # re19_reddit_loader = RumorEvalReddit19(config.re19_root)
    # re19_reddit_dataset = re19_reddit_loader.load()
    # re19_reddit_dataset.export_full(os.path.join(config.re19_root, "re19_reddit.csv"))


def load_fnn(config, news_label):
    fnn_loader = FakeNewsNetDatasetLoader(config.fnn_root)
    fnn_dataset = fnn_loader.load(clean=False, news_label=news_label)
    fnn_dataset.export_full(os.path.join(config.fnn_root, "fnn_{}_uncleaned.csv".format(news_label)))
    fnn_dataset_cleaned = fnn_loader.load(clean=True, news_label=news_label)
    fnn_dataset_cleaned.export_full(os.path.join(config.fnn_root, "fnn_{}_cleaned.csv".format(news_label)))


def combine_stance_relation_all():
    combine_stance_relation("datasets/annotated/fnn/cleaned/")
    combine_stance_relation("datasets/annotated/fnn/uncleaned/")
    combine_stance_relation("datasets/annotated/csi/cleaned/")
    combine_stance_relation("datasets/annotated/csi/uncleaned/")


def analyse_fnn(config):
    fnn_loader = FakeNewsNetDatasetLoader(config.fnn_root)
    fnn_loader.export_mentioned_urls(os.path.join(config.fnn_root, "mentioned_urls.csv"),
                                     os.path.join(config.fnn_root, "mentioned_urls_freq.csv"))
    # fnn_loader.export_source_urls_analysis(os.path.join(config.fnn_root, "source_urls.csv"))


def analyse(config):
    analyse_fnn(config)


def preprocess_stance_fnc(config):
    fnc_loader = FncLoader(config.fnc_root)
    fnc_dataset = fnc_loader.load()
    fnc_path = os.path.join(config.fnc_root, "data.tsv")
    fnc_dataset.export_full(fnc_path, delimiter="\t")
    fnc_train_path, fnc_test_path = process_text_classification(fnc_path, os.path.dirname(fnc_path))
    cross_val(fnc_train_path, os.path.dirname(fnc_train_path), num_folds=10)


def preprocess_stance_fnn(config):
    cleaned_output_path, uncleaned_output_path = process_annotated_datasets(config)
    cleaned_train_path, cleaned_test_path = \
        process_text_classification(cleaned_output_path, os.path.dirname(cleaned_output_path))
    uncleaned_train_path, uncleaned_test_path = \
        process_text_classification(uncleaned_output_path, os.path.dirname(uncleaned_output_path))
    cross_val(cleaned_train_path, os.path.dirname(cleaned_train_path), num_folds=10)
    cross_val(uncleaned_train_path, os.path.dirname(uncleaned_train_path), num_folds=10)


def preprocess(dataset_name, config):
    if dataset_name == "stance_fnn":
        preprocess_stance_fnn(config)
    elif dataset_name == "stance_fnc":
        preprocess_stance_fnc(config)
    else:
        raise ValueError("Unrecognized stance dataset {}".format(dataset_name))

    # load_fnc_full(config)
    # load_fnc_split(config)
    # load_re17(config)
    #
    # (config)
    # load_fnc_relation_full(config)
    # load_fnc_relation_split(config)
    # load_fnn(config, "fake")
    # load_fnn(config, "real")
    # eval_stance("datasets/fnc_full/stance.tsv", "datasets/fnc_full/test.tsv")
    # eval_stance("datasets/fnc_relation_full/relation.tsv", "datasets/fnc_relation_full/test.tsv")
    # eval_stance("datasets/fnc_split/stance.tsv", "datasets/fnc_split/test.tsv")
    # eval_stance("datasets/fnc_relation_split/relation.tsv", "datasets/fnc_relation_split/test.tsv")
    # eval_stance("datasets/rumor_eval_19/only_twitter/fold_1/stance.tsv",
    #             "datasets/rumor_eval_19/only_twitter/fold_1/dev.tsv")
    # process_csi_dataset(config, tweet_limit_per_event=50)
    # process_text_classification("datasets/annotated/fnn/cleaned/data.tsv", "datasets/annotated/fnn/cleaned/")
    # process_text_classification("datasets/annotated/fnn/uncleaned/data.tsv", "datasets/annotated/fnn/uncleaned/")
    # # combine_stance_relation_all()
    # eval_stance("datasets/annotated/csi/cleaned/stance_relation.tsv", "datasets/annotated/csi/cleaned/test.tsv")
    # load_re19(config)


def eval_stance(stance_pred_path, stance_truth_path):
    preds = utils.read_csv(stance_pred_path, delimiter="\t")
    truths = utils.read_csv(stance_truth_path, delimiter="\t")
    y_preds = [row[1] for row in preds]
    y_truths = [row[3] for row in truths]

    Evaluator.evaluate_multi_class(y_preds, y_truths)


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
    preprocess(CONFIGS.dataset_name, sd_config)
    # experiment_summary(sd_config)
    # analyse(sd_config)
