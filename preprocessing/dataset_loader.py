import csv
import os
import urllib3
import utils

from preprocessing.dataset import StanceDataset


class BaseDatasetLoader(object):

    def __init__(self, info_every=10):
        self.info_every = info_every

    def load(self):
        return StanceDataset([], [])


class FakeNewsNetDatasetLoader(BaseDatasetLoader):

    DEFAULT_STANCE = "comment"
    DATASETS = ["politifact"]

    def __init__(self, fnn_root, info_every=10):
        super().__init__(info_every)
        self.fnn_root = fnn_root

    def load(self, clean=True):
        feature_set, label_set = [], []

        i = 0
        for dataset in self.DATASETS:
            dataset_dir = os.path.join(self.fnn_root, dataset)
            for news_label in os.listdir(dataset_dir):
                news_label_dir = os.path.join(dataset_dir, news_label)
                for news_id in os.listdir(news_label_dir):
                    news_dir = os.path.join(news_label_dir, news_id)
                    tweet_dir = os.path.join(news_dir, "tweets")
                    news_content_path = os.path.join(news_dir, "news content.json")
                    news_content = utils.read_json(news_content_path)
                    news_title = news_content["title"] if "title" in news_content else ""
                    news_description = news_content["meta_data"]["description"] \
                        if "description" in news_content["meta_data"] else ""
                    if clean:
                        news_title = utils.clean_tweet_text(news_title)
                        news_description = utils.clean_tweet_text(news_description)

                    for tweet_id in os.listdir(tweet_dir):
                        if i % self.info_every == 0:
                            print("Loaded {} Fake News Net tweets".format(str(i)))
                        tweet_path = os.path.join(tweet_dir, tweet_id)
                        tweet_content = utils.read_json(tweet_path)
                        tweet_text = utils.clean_tweet_text(tweet_content["text"]) if clean else tweet_content["text"]
                        if len(news_title) > 0:
                            clean_tweet_text = utils.clean_stance_target(tweet_text, news_title) \
                                if clean else tweet_text
                            feature_set.append([clean_tweet_text, news_title])
                            label_set.append(self.DEFAULT_STANCE)
                        if len(news_description) > 0:
                            clean_tweet_text = utils.clean_stance_target(tweet_text, news_description) \
                                if clean else tweet_text
                            feature_set.append([clean_tweet_text, news_description])
                            label_set.append(self.DEFAULT_STANCE)
                        i += 1

        return StanceDataset(feature_set, label_set)


class RumorEval17(BaseDatasetLoader):
    HEADLINE_HEADER = ["url_id", "full_url", "headline", "media", "clean"]

    def __init__(self, re_root, info_every=10):
        super().__init__(info_every)
        self.re_root = re_root
        self.re_data = os.path.join(self.re_root, "rumoureval-data")
        self.traindev = os.path.join(self.re_root, "traindev")
        self.headline_csv = os.path.join(self.re_root, "headlines.csv")

    def load_labels(self):
        re_train_label_path = os.path.join(self.traindev, "rumoureval-subtaskA-train.json")
        re_dev_label_path = os.path.join(self.traindev, "rumoureval-subtaskA-dev.json")
        train_labels = utils.read_json(re_train_label_path)
        dev_labels = utils.read_json(re_dev_label_path)
        raw_label_map = {**train_labels, **dev_labels}
        return {str(k): v for k, v in raw_label_map.items()}  # convert tweet id from int to string

    def load_headlines(self):
        headline_map = {}
        with open(self.headline_csv, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar='"')
            for row in csv_reader:
                url_id, full_url, headline, news_media, cleaned = row
                if cleaned == "1":
                    headline_map[url_id] = headline
        return headline_map

    def crawl_headlines(self):
        headline_map = {}
        headline_crawler = utils.HeadlineCrawler()

        i = 0
        for topic in os.listdir(self.re_data):
            topic_path = os.path.join(self.re_data, topic)
            for discourse_id in os.listdir(topic_path):
                discourse_id_path = os.path.join(topic_path, discourse_id)
                url_data = os.path.join(discourse_id_path, "urls.dat")

                with open(url_data, "r", encoding="utf-8") as f:
                    csv_reader = csv.reader(f, delimiter="\t", quotechar='"')
                    for row in csv_reader:
                        if i % self.info_every == 0:
                            print("Loaded {} rumor eval 17 headlines".format(str(i)))
                        url_id, shortened_url, full_url = row[0], row[1], row[2]
                        try:
                            if url_id not in headline_map:
                                headline, news_media = headline_crawler.crawl_url_title(full_url)
                                if url_id not in headline_map:
                                    headline_map[url_id] = (full_url, headline, news_media)
                            i += 1
                        except urllib3.exceptions.HTTPError:
                            continue

        # by default all rows are clean
        headline_content = [[url_id, full_url, headline, news_media, "1"]
                            for url_id, (full_url, headline, news_media) in headline_map.items()]
        utils.write_csv(headline_content, RumorEval17.HEADLINE_HEADER, self.headline_csv)

    @staticmethod
    def convert_stance(raw_stance):
        stances_map = {
            "support": "support",
            "deny": "deny",
            "comment": "comment"
        }

        if raw_stance in stances_map.keys():
            return stances_map[raw_stance]
        raise ValueError("Unsupported stance {}".format(raw_stance))

    @staticmethod
    def load_tweet_folder(tweet_folder_path):
        tweet_map = {}
        for tweet_path in os.listdir(tweet_folder_path):
            tweet_full_path = os.path.join(tweet_folder_path, tweet_path)
            tweet_json = utils.read_json(tweet_full_path)
            tweet_map[str(tweet_json["id"])] = utils.clean_tweet_text(tweet_json["text"])
        return tweet_map

    @staticmethod
    def propagate_headline_label(tweet_stance, reply_stance):
        if (tweet_stance == "support" and reply_stance == "support") \
                or (tweet_stance == "deny" and reply_stance == "deny"):
            return "support"
        if (tweet_stance == "support" and reply_stance == "deny") \
                or (tweet_stance == "deny" and reply_stance == "support"):
            return "deny"
        raise ValueError("Tweet stance {} and reply stance {} cannot be propagated".format(tweet_stance, reply_stance))

    @staticmethod
    def annotate_tweet(tweet_structure, label_map, headlines, feature_set,
                       label_set, prop_headline_label_map, full_tweet_map):
        for tweet, replies in tweet_structure.items():
            if type(replies) is not dict:
                continue
            for reply in replies.keys():
                if tweet not in full_tweet_map:
                    print("Tweet {} does not have any text".format(tweet))
                    continue
                if reply not in full_tweet_map:
                    print("Tweet {} does not have any text".format(reply))
                    continue
                if reply not in label_map:
                    print("Tweet {} does not have any label".format(reply))
                    continue
                tweet_text = full_tweet_map[tweet]
                reply_text = full_tweet_map[reply]
                try:
                    reply_stance = RumorEval17.convert_stance(label_map[reply])
                    feature_set.append([reply_text, tweet_text])
                    label_set.append(reply_stance)
                    if tweet in prop_headline_label_map:
                        reply_headline_label = RumorEval17.propagate_headline_label(
                            prop_headline_label_map[tweet], reply_stance)
                        prop_headline_label_map[reply] = reply_headline_label
                        for headline in headlines:
                            feature_set.append([reply_text, headline])
                            label_set.append(reply_headline_label)
                except ValueError as e:
                    print(str(e))
            RumorEval17.annotate_tweet(replies, label_map, headlines, feature_set,
                                       label_set, prop_headline_label_map, full_tweet_map)

    def load(self):
        if not os.path.isfile(self.headline_csv):
            print("Crawl new headline data to {}".format(self.headline_csv))
            self.crawl_headlines()
        headline_map = self.load_headlines()
        label_map = self.load_labels()
        feature_set, label_set = [], []

        i = 0
        for topic in os.listdir(self.re_data):
            topic_path = os.path.join(self.re_data, topic)
            for discourse_id in os.listdir(topic_path):
                if i % self.info_every == 0:
                    print("Loaded {} rumor eval 17 discourses".format(str(i)))
                discourse_id_path = os.path.join(topic_path, discourse_id)
                reply_folder_path = os.path.join(discourse_id_path, "replies")
                source_folder_path = os.path.join(discourse_id_path, "source-tweet")
                reply_map = self.load_tweet_folder(reply_folder_path)
                source_map = self.load_tweet_folder(source_folder_path)
                full_tweet_map = {**reply_map, **source_map}
                url_data = os.path.join(discourse_id_path, "urls.dat")
                tweet_structure = utils.read_json(os.path.join(discourse_id_path, "structure.json"))
                headlines = []
                with open(url_data, "r", encoding="utf-8") as f:
                    csv_reader = csv.reader(f, delimiter="\t", quotechar='"')
                    for row in csv_reader:
                        url_id, shortened_url, full_url = row[0], row[1], row[2]
                        if url_id in headline_map:
                            headlines.append(headline_map[url_id])
                        else:
                            print("Url {} not found".format(url_id))
                prop_headline_label_map = {}
                for source, source_text in source_map.items():
                    if source not in label_map:
                        print("Tweet {} does not have any label".format(source))
                        continue
                    try:
                        source_label = self.convert_stance(label_map[source])
                        for headline in headlines:
                            feature_set.append([source_text, headline])
                            label_set.append(source_label)
                            prop_headline_label_map[source] = source_label
                    except ValueError as e:
                        print(str(e))
                self.annotate_tweet(tweet_structure, label_map, headlines, feature_set,
                                    label_set, prop_headline_label_map, full_tweet_map)
            i += 1
        return StanceDataset(feature_set, label_set)


class FncLoader(BaseDatasetLoader):

    def __init__(self, fnc_root, info_every=1000):
        super().__init__(info_every)
        self.fnc_root = fnc_root
        self.fnc_train_bodies = os.path.join(fnc_root, "train_bodies.csv")
        self.fnc_train_stances = os.path.join(fnc_root, "train_stances.csv")

    def load_bodies(self):
        bodies_map = {}
        with open(self.fnc_train_bodies, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar='"')
            for i, row in enumerate(csv_reader):
                if i % self.info_every == 0:
                    print("Loaded {} rows of fnc bodies".format(str(i)))
                if i == 0:
                    print("Skipping header " + str(row))
                else:
                    body_id, body_content = row[0], row[1]
                    bodies_map[body_id] = body_content
        return bodies_map

    @staticmethod
    def convert_stance(raw_stance):
        stances_map = {
            "agree": "support",
            "disagree": "deny",
            "discuss": "comment",
            "unrelated": "unrelated"
        }

        if raw_stance in stances_map.keys():
            return stances_map[raw_stance]
        raise ValueError("Unsupported stance {}".format(raw_stance))

    def load_full(self):
        bodies_map = self.load_bodies()
        feature_set, label_set = [], []

        with open(self.fnc_train_stances, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar='"')
            for i, row in enumerate(csv_reader):
                if i % self.info_every == 0:
                    print("Loaded {} rows of fnc stances".format(str(i)))
                if i == 0:
                    print("Skipping header " + str(row))
                else:
                    try:
                        headline, body_id, stance = row[0], row[1], self.convert_stance(row[2])
                        body_content = bodies_map[body_id]
                        feature_set.append([body_content, headline])
                        label_set.append(stance)
                    except ValueError as e:
                        continue
                        # print(str(e))

        return feature_set, label_set

    def load(self):
        feature_set, label_set = self.load_full()
        cleaned_feature_set = []
        for i, feature in enumerate(feature_set):
            source, target = feature[0], feature[1]
            cleaned_feature_set.append([source.replace("\n", " "), target])
        return StanceDataset(cleaned_feature_set, label_set)

    def load_split(self):
        feature_set, label_set = self.load_full()
        split_feature_set, split_label_set = [], []
        for i, feature in enumerate(feature_set):
            label = label_set[i]
            source, target = feature[0], feature[1]
            for paragraph in source.splitlines():
                if len(paragraph.strip().rstrip()) > 0:
                    split_feature_set.append([paragraph, target])
                    split_label_set.append(label)

        return StanceDataset(split_feature_set, split_label_set)


class FncRelationLoader(FncLoader):
    def __init__(self, fnc_root, info_every=1000):
        super(FncRelationLoader, self).__init__(fnc_root, info_every)

    @staticmethod
    def convert_stance(raw_stance):
        stances_map = {
            "agree": "related",
            "disagree": "related",
            "discuss": "related",
            "unrelated": "unrelated"
        }

        if raw_stance in stances_map.keys():
            return stances_map[raw_stance]
        raise ValueError("Unsupported stance {}".format(raw_stance))