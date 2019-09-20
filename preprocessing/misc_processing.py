import numpy as np
from sklearn.model_selection import KFold
import os
import pandas as pd
from preprocessing.dataset import StanceDataset
from sklearn.model_selection import train_test_split
import utils


def process_fnn(fnn_csv_path, output_dir, num_folds=10):
    csv_content = utils.read_csv(fnn_csv_path, delimiter="\t")
    idxs, features, labels = [], [], []
    for sample in csv_content:
        idxs.append(sample[0])
        features.append([sample[1], sample[2]])
        labels.append(sample[3].rstrip())
    stance_dataset = StanceDataset(features, labels, idxs=idxs)
    stance_dataset.export_cross_eval(output_dir, num_folds)


def combine_stance_relation(root_path):
    stance_path = os.path.join(root_path, "stance.tsv")
    relation_path = os.path.join(root_path, "relation.tsv")

    stance_data = utils.read_csv(stance_path, delimiter="\t")
    relation_data = utils.read_csv(relation_path, delimiter="\t")

    assert len(stance_data) == len(relation_data)
    combined_data = []

    for i, row in enumerate(stance_data):
        stance = stance_data[i][1]
        relation = relation_data[i][1]
        final_stance = "unrelated" if relation == "unrelated" else stance
        combined_data.append([i, final_stance])

    utils.write_csv(combined_data, ["index", "source", "target", "stance"],
                    os.path.join(root_path, "stance_relation.tsv"), delimiter="\t")


def process_csi_dataset(config, tweet_limit_per_event=None):
    raw_twitter_path = os.path.join(config.csi_root, "Twitter.txt")
    url_place_holder = "PLEASE_ENTER_URL"
    title_place_holder = "PLEASE_ENTER_TITLE"
    fake_twitter_content = []
    real_twitter_content = []
    with open(raw_twitter_path, "r") as f:
        for line in f.readlines():
            tokens = line.split("\t")
            event_id, label, tweets = tokens[0], tokens[1], tokens[2].split(" ")
            if tweet_limit_per_event is not None:
                tweets = tweets[:tweet_limit_per_event]
            if label == "label:0":
                real_twitter_content.append([event_id.replace("eid:", ""), url_place_holder, title_place_holder,
                                             "\t".join(tweets).replace("\n", "")])
            elif label == "label:1":
                fake_twitter_content.append([event_id.replace("eid:", ""), url_place_holder, title_place_holder,
                                             "\t".join(tweets).replace("\n", "")])
            else:
                print("Unrecognized label {}".format(label))
    fnn_header = ["id", "news_url", "title", "tweet_ids"]
    utils.write_csv(fake_twitter_content, fnn_header, os.path.join(config.csi_root, "twitter_fake.csv"))
    utils.write_csv(real_twitter_content, fnn_header, os.path.join(config.csi_root, "twitter_real.csv"))


def process_annotated_datasets(config, label_type="stance"):
    annotated_root = os.path.join("datasets", "fnn")
    all_batch_dir = os.path.join(annotated_root, "batches")
    fnn_cleaned_dfs, fnn_uncleaned_dfs = [], []
    for batch in os.listdir(all_batch_dir):
        batch_dir = os.path.join(all_batch_dir, batch)
        if label_type == "stance":
            fnn_fake_cleaned = process_stance_annotated_dataset(os.path.join(batch_dir, "fnn_fake_uncleaned.xlsx"))
            fnn_real_cleaned = process_stance_annotated_dataset(os.path.join(batch_dir, "fnn_real_uncleaned.xlsx"))
            fnn_fake_uncleaned = process_stance_annotated_dataset(os.path.join(batch_dir, "fnn_fake_uncleaned.xlsx"),
                                                                  cleaning=False)
            fnn_real_uncleaned = process_stance_annotated_dataset(os.path.join(batch_dir, "fnn_real_uncleaned.xlsx"),
                                                                  cleaning=False)
        elif label_type == "sentiment":
            fnn_fake_cleaned = process_sentiment_annotated_dataset(os.path.join(batch_dir, "fnn_fake_uncleaned.xlsx"))
            fnn_real_cleaned = process_sentiment_annotated_dataset(os.path.join(batch_dir, "fnn_real_uncleaned.xlsx"))
            fnn_fake_uncleaned = process_sentiment_annotated_dataset(os.path.join(batch_dir, "fnn_fake_uncleaned.xlsx"),
                                                                     cleaning=False)
            fnn_real_uncleaned = process_sentiment_annotated_dataset(os.path.join(batch_dir, "fnn_real_uncleaned.xlsx"),
                                                                     cleaning=False)
        else:
            raise ValueError("Unrecognized label type {}".format(label_type))
        fnn_cleaned_dfs += [fnn_fake_cleaned, fnn_real_cleaned]
        fnn_uncleaned_dfs += [fnn_fake_uncleaned, fnn_real_uncleaned]
    fnn_cleaned = pd.concat(fnn_cleaned_dfs, sort=False)
    fnn_uncleaned = pd.concat(fnn_uncleaned_dfs, sort=False)
    if label_type == "stance":
        fnn_cleaned = post_process_stance_annotated_dataset(fnn_cleaned)
        fnn_uncleaned = post_process_stance_annotated_dataset(fnn_uncleaned)
    elif label_type == "sentiment":
        fnn_cleaned = post_process_sentiment_annotated_dataset(fnn_cleaned)
        fnn_uncleaned = post_process_sentiment_annotated_dataset(fnn_uncleaned)
    else:
        raise ValueError("Unrecognized label type {}".format(label_type))
    cleaned_output_path = os.path.join(annotated_root, label_type, "cleaned", "{}.tsv".format(label_type))
    uncleaned_output_path = os.path.join(annotated_root, label_type, "uncleaned", "{}.tsv".format(label_type))
    fnn_cleaned.to_csv(utils.io.ensure_path(cleaned_output_path), index=False, sep='\t')
    fnn_uncleaned.to_csv(utils.io.ensure_path(uncleaned_output_path), index=False, sep='\t')
    return cleaned_output_path, uncleaned_output_path


def process_sentiment_annotated_dataset(annotated_path, cleaning=True):
    print("Process sentiment annotated dataset at {}".format(annotated_path))
    annotated_df = pd.read_excel(pd.ExcelFile(annotated_path))
    filtered_clean_df = annotated_df[annotated_df["clean"] == 1]
    filtered_comment_df = filtered_clean_df[filtered_clean_df["stance"] == "comment"]
    filtered_comment_df["source"] = filtered_comment_df["source"] \
        .map(lambda x: utils.simple_clean(x))
    if cleaning:
        filtered_comment_df["source"] = filtered_comment_df["source"] \
            .map(lambda x: utils.clean_tweet_text(x))
    filtered_comment_df = filtered_comment_df[["source", "sentiment"]]
    return filtered_comment_df


def process_stance_annotated_dataset(annotated_path, cleaning=True):
    print("Process stance annotated dataset at {}".format(annotated_path))
    annotated_df = pd.read_excel(pd.ExcelFile(annotated_path))
    filtered_report_df = annotated_df.set_index("stance")
    try:
        filtered_report_df = filtered_report_df.drop("report", axis=0).dropna(how="all")
    except KeyError as e:
        print(str(e))
    filtered_report_df = filtered_report_df.reset_index()
    filtered_clean_df = filtered_report_df[filtered_report_df["clean"] == 1]
    filtered_clean_df["source"] = filtered_clean_df["source"] \
        .map(lambda x: utils.simple_clean(x))
    filtered_clean_df["target"] = filtered_clean_df["target"] \
        .map(lambda x: utils.simple_clean(x))
    if cleaning:
        filtered_clean_df["source"] = filtered_clean_df["source"] \
            .map(lambda x: utils.clean_tweet_text(x))
    filtered_clean_df = filtered_clean_df[["source", "target", "stance"]]
    return filtered_clean_df


def post_process_stance_annotated_dataset(df):
    final_df = df.drop_duplicates(subset=["source", "target"])
    final_df['idx'] = range(1, len(final_df) + 1)
    final_df = final_df[["idx", "source", "target", "stance"]]
    return final_df


def post_process_sentiment_annotated_dataset(df):
    final_df = df.drop_duplicates(subset=["source"])
    final_df['idx'] = range(1, len(final_df) + 1)
    final_df = final_df[["idx", "source", "sentiment"]]
    return final_df


def process_text_classification(stance_path, stance_dataset_dir, label_type="stance", dataset_name="data"):
    stance_dataset = utils.read_csv(stance_path, delimiter="\t")
    content_out = []
    for row in stance_dataset:
        if label_type == "stance":
            if len(row[1].strip().rstrip()) > 0 and len(row[2].strip().rstrip()) > 0:
                content_out.append([row[0], row[1], row[2], row[3].rstrip()])
        elif label_type == "sentiment":
            if len(row[1].strip().rstrip()) > 0:
                content_out.append([row[0], row[1], row[2].rstrip()])
    train, test = train_test_split(content_out, test_size=0.05, random_state=9)
    train_path = os.path.join(stance_dataset_dir, "{}_all.train".format(dataset_name))
    test_path = os.path.join(stance_dataset_dir, "{}.test".format(dataset_name))
    utils.write_csv(train, None, train_path, delimiter="\t")
    utils.write_csv(test, None, test_path, delimiter="\t")
    return train_path, test_path


def cross_val(input_path, output_dir, num_folds, dataset_name="data"):
    kf = KFold(n_splits=num_folds)
    data = np.array(utils.load_text_as_list(input_path))
    fold = 1
    for train_index, test_index in kf.split(data):
        fold_dir = os.path.join(output_dir, "fold_{}".format(fold))
        print("Creating fold {} at {}".format(fold, output_dir))
        data_train, data_test = data[train_index], data[test_index]
        utils.save_list_as_text(data_train, utils.ensure_path(os.path.join(fold_dir, "{}.train".format(dataset_name))))
        utils.save_list_as_text(data_test, utils.ensure_path(os.path.join(fold_dir, "{}.val".format(dataset_name))))
        fold += 1
