import os
import pandas as pd
import utils


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
    csi_fake_cleaned = process_annotated_dataset(os.path.join(annotated_root, "csi_fake_cleaned.xlsx"))
    csi_real_cleaned = process_annotated_dataset(os.path.join(annotated_root, "csi_real_cleaned.xlsx"))
    csi_fake_uncleaned = process_annotated_dataset(os.path.join(annotated_root, "csi_fake_uncleaned.xlsx"))
    csi_real_uncleaned = process_annotated_dataset(os.path.join(annotated_root, "csi_real_uncleaned.xlsx"))
    csi_cleaned = pd.concat([csi_fake_cleaned, csi_real_cleaned], sort=False)
    csi_uncleaned = pd.concat([csi_fake_uncleaned, csi_real_uncleaned], sort=False)
    fnn_fake_cleaned = process_annotated_dataset(os.path.join(annotated_root, "fnn_fake_cleaned.xlsx"))
    fnn_real_cleaned = process_annotated_dataset(os.path.join(annotated_root, "fnn_real_cleaned.xlsx"))
    fnn_fake_uncleaned = process_annotated_dataset(os.path.join(annotated_root, "fnn_fake_uncleaned.xlsx"))
    fnn_real_uncleaned = process_annotated_dataset(os.path.join(annotated_root, "fnn_real_uncleaned.xlsx"))
    fnn_cleaned = pd.concat([fnn_fake_cleaned, fnn_real_cleaned], sort=False)
    fnn_uncleaned = pd.concat([fnn_fake_uncleaned, fnn_real_uncleaned], sort=False)
    csi_cleaned.to_csv(os.path.join(annotated_root, "csi", "cleaned", "test.tsv"), index=False, sep='\t')
    csi_uncleaned.to_csv(os.path.join(annotated_root, "csi", "uncleaned", "test.tsv"), index=False, sep='\t')
    fnn_cleaned.to_csv(os.path.join(annotated_root, "fnn", "cleaned", "test.tsv"), index=False, sep='\t')
    fnn_uncleaned.to_csv(os.path.join(annotated_root, "fnn", "uncleaned", "test.tsv"), index=False, sep='\t')


def process_annotated_dataset(annotated_path):
    print("Process annotated dataset at {}".format(annotated_path))
    # annotated_df = pd.read_csv(annotated_path, encoding="ISO-8859-1")
    annotated_df = pd.read_excel(pd.ExcelFile(annotated_path))
    filtered_report_df = annotated_df.set_index("stance")
    print(filtered_report_df.describe())
    filtered_report_df = filtered_report_df.drop("report", axis=0).dropna(how="all")
    filtered_report_df = filtered_report_df.reset_index()
    print(filtered_report_df.columns)
    filtered_report_df["source"] = filtered_report_df["source"].map(lambda x: str(x).replace("\n", " "))
    filtered_report_df["target"] = filtered_report_df["target"].map(lambda x: str(x).replace("\n", " "))
    filtered_report_df = filtered_report_df[["index", "source", "target", "stance"]]
    return filtered_report_df
