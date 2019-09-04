import os
import pandas as pd
from preprocessing import StanceDataset
from sklearn.model_selection import train_test_split
import utils


def process_fnn(fnn_csv_path, output_dir, num_folds=10):
    csv_content = utils.read_csv(fnn_csv_path, delimiter="\t")
    features, labels = [], []
    for sample in csv_content:
        features.append([sample[1], sample[2]])
        labels.append([sample[3]])
    stance_dataset = StanceDataset(features, labels)
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
            # print(tokens)
            # break
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


def process_annotated_datasets(config):
    annotated_root = os.path.join("datasets", "annotated")
    all_batch_dir = os.path.join(annotated_root, "batches")
    fnn_cleaned_dfs, fnn_uncleaned_dfs = [], []
    for batch in os.listdir(all_batch_dir):
        batch_dir = os.path.join(all_batch_dir, batch)
        fnn_fake_cleaned = process_annotated_dataset(os.path.join(batch_dir, "fnn_fake_cleaned.xlsx"))
        fnn_real_cleaned = process_annotated_dataset(os.path.join(batch_dir, "fnn_real_cleaned.xlsx"))
        fnn_fake_uncleaned = process_annotated_dataset(os.path.join(batch_dir, "fnn_fake_uncleaned.xlsx"))
        fnn_real_uncleaned = process_annotated_dataset(os.path.join(batch_dir, "fnn_real_uncleaned.xlsx"))
        fnn_cleaned_dfs += [fnn_fake_cleaned, fnn_real_cleaned]
        fnn_uncleaned_dfs += [fnn_fake_uncleaned, fnn_real_uncleaned]
    fnn_cleaned = pd.concat(fnn_cleaned_dfs, sort=False)
    fnn_uncleaned = pd.concat(fnn_uncleaned_dfs, sort=False)
    fnn_cleaned = post_process_annotated_dataset(fnn_cleaned)
    fnn_uncleaned = post_process_annotated_dataset(fnn_uncleaned)
    fnn_cleaned.to_csv(os.path.join(annotated_root, "fnn", "cleaned", "data.tsv"), index=False, sep='\t')
    fnn_uncleaned.to_csv(os.path.join(annotated_root, "fnn", "uncleaned", "data.tsv"), index=False, sep='\t')


def process_annotated_dataset(annotated_path):
    print("Process annotated dataset at {}".format(annotated_path))
    annotated_df = pd.read_excel(pd.ExcelFile(annotated_path))
    filtered_report_df = annotated_df.set_index("stance")
    print(filtered_report_df.describe())
    filtered_report_df = filtered_report_df.drop("report", axis=0).dropna(how="all")
    filtered_report_df = filtered_report_df.reset_index()
    filtered_clean_df = filtered_report_df[filtered_report_df["clean"] == 0]
    print(filtered_clean_df.columns)
    filtered_clean_df["source"] = filtered_clean_df["source"] \
        .map(lambda x: utils.clean_tweet_text(str(x).lower()).replace("\n", " "))
    filtered_clean_df["target"] = filtered_clean_df["target"] \
        .map(lambda x: utils.clean_tweet_text(str(x).lower()).replace("\n", " "))
    filtered_clean_df = filtered_clean_df[["index", "source", "target", "stance"]]
    return filtered_clean_df


def post_process_annotated_dataset(df):
    final_df = df.drop_duplicates(subset=["source", "target"])
    return final_df


def process_text_classification(stance_path, stance_dataset_dir):
    stance_dataset = utils.read_csv(stance_path)
    stance_map = {
        "support": 0,
        "deny": 1,
        "comment": 2,
        "unrelated": 3
    }
    content_out = [[row[1], row[2], stance_map[row[3]]] for row in stance_dataset]
    train, test = train_test_split(content_out, test_size=0.1, random_state=9)
    utils.write_csv(train, None, os.path.join(stance_dataset_dir, "data.train"), delimiter="\t")
    utils.write_csv(test, None, os.path.join(stance_dataset_dir, "data.test"), delimiter="\t")
