from utils.io import read_csv, write_csv


def raw_label_map(raw_label):
    if raw_label in ["(3, 2)", "(4, 1)", "(5, 0)"]:
        return "paraphrases"
    elif raw_label in ["(1, 4)", "(0, 5)"]:
        return "non_paraphrases"
    elif raw_label in ["(2, 3)"]:
        return "debatable"
    raise ValueError("Unrecognized label {}".format(raw_label))


def process_tweet_paraphrase(input_path, output_path):
    raw_tweet_dataset = read_csv(input_path, load_header=True, delimiter="\t")
    tweet_dataset = [[idx, row[2], row[3], raw_label_map(row[4])] for idx, row in enumerate(raw_tweet_dataset)]
    header = ["index", "sent1", "sent2", "label"]
    write_csv(tweet_dataset, header, output_path, delimiter="\t")
