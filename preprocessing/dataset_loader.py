from collections import Counter
import csv
import urllib3

from preprocessing.dataset import *


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

    def load_mentioned_urls(self, news_label="fake"):
        mentioned_urls_dict, mentioned_urls_corpus = {}, []
        print("Load mentioned urls")
        i = 0

        for dataset in self.DATASETS:
            dataset_dir = os.path.join(self.fnn_root, dataset)
            news_label_dir = os.path.join(dataset_dir, news_label)
            for news_id in os.listdir(news_label_dir):
                mentioned_urls_dict[news_id] = set()
                unique_mentioned_urls = set()
                news_dir = os.path.join(news_label_dir, news_id)
                tweet_dir = os.path.join(news_dir, "tweets")
                print(tweet_dir)
                if not os.path.isdir(tweet_dir):
                    continue
                for tweet_id in os.listdir(tweet_dir):
                    if i % self.info_every == 0:
                        print("Loaded {} Fake News Net tweets".format(str(i)))
                    tweet_path = os.path.join(tweet_dir, tweet_id)
                    tweet_content = utils.read_json(tweet_path)
                    if "entities" not in tweet_content or "urls" not in tweet_content["entities"] \
                            or len(tweet_content["entities"]["urls"]) == 0:
                        continue
                    for url_obj in tweet_content["entities"]["urls"]:
                        expanded_url = url_obj["expanded_url"]
                        if len(expanded_url) > 0:
                            home_expanded_url = utils.extract_home_url(expanded_url)
                            mentioned_urls_dict[news_id].add(home_expanded_url)
                            unique_mentioned_urls.add(home_expanded_url)
                    i += 1
                mentioned_urls_corpus += list(unique_mentioned_urls)

        mentioned_urls = [[k, " ".join(v)] for k, v in mentioned_urls_dict.items()]
        print("Size of mentioned urls " + str(len(mentioned_urls)))
        return mentioned_urls, mentioned_urls_corpus

    def export_mentioned_urls(self, mentioned_urls_path, mentioned_urls_freq_path):
        fake_mentioned_urls, fake_mentioned_urls_corpus = self.load_mentioned_urls(news_label="fake")
        real_mentioned_urls, real_mentioned_urls_corpus = self.load_mentioned_urls(news_label="real")
        mentioned_urls = fake_mentioned_urls + real_mentioned_urls
        # mentioned_urls_corpus = fake_mentioned_urls_corpus + real_mentioned_urls_corpus
        fake_cnt, real_cnt = Counter(), Counter()
        for url in fake_mentioned_urls_corpus:
            fake_cnt[url] += 1
        for url in real_mentioned_urls_corpus:
            real_cnt[url] += 1
        cnt = fake_cnt + real_cnt
        mentioned_urls_freq_content = []
        mentioned_urls_freq_header = ["mentioned_url", "frequency", "label"]
        mentioned_urls_header = ["event", "mentioned_urls"]
        for url, frequency in cnt.most_common():
            fake_url_cnt = fake_cnt[url] if url in fake_cnt else 0
            real_url_cnt = real_cnt[url] if url in real_cnt else 0
            if fake_url_cnt > real_url_cnt:
                label = "fake"
            elif fake_url_cnt < real_url_cnt:
                label = "real"
            else:
                label = "tie"
            mentioned_urls_freq_content.append([url, str(frequency), label])
        utils.write_csv(mentioned_urls_freq_content, mentioned_urls_freq_header, mentioned_urls_freq_path)
        utils.write_csv(mentioned_urls, mentioned_urls_header, mentioned_urls_path)

    def export_source_urls_analysis(self, path):
        fake_source_urls = self.load_source_urls(news_label="fake")
        real_source_urls = self.load_source_urls(news_label="real")
        source_urls = fake_source_urls + real_source_urls
        cnt = Counter()
        for url in source_urls:
            cnt[url] += 1
        content = []
        header = ["source_url", "frequency"]
        for source_url, frequency in cnt.most_common():
            content.append([source_url, str(frequency)])
        utils.write_csv(content, header, path)

    def load_source_urls(self, news_label="fake"):
        source_urls = []

        for dataset in self.DATASETS:
            dataset_dir = os.path.join(self.fnn_root, dataset)
            news_label_dir = os.path.join(dataset_dir, news_label)
            for news_id in os.listdir(news_label_dir):
                news_dir = os.path.join(news_label_dir, news_id)
                news_content_path = os.path.join(news_dir, "news content.json")
                if not os.path.isfile(news_content_path):
                    continue
                news_content = utils.read_json(news_content_path)
                source_url = news_content["url"]
                source_home_url = utils.extract_home_url(source_url)
                source_urls.append(source_home_url)

        return source_urls


    def load(self, clean=True, news_label="fake"):
        feature_set, label_set = [], []

        i = 0
        for dataset in self.DATASETS:
            dataset_dir = os.path.join(self.fnn_root, dataset)
            news_label_dir = os.path.join(dataset_dir, news_label)
            for news_id in os.listdir(news_label_dir):
                if i > 10000:
                    break
                news_dir = os.path.join(news_label_dir, news_id)
                tweet_dir = os.path.join(news_dir, "tweets")
                news_content_path = os.path.join(news_dir, "news content.json")
                if not os.path.isfile(news_content_path):
                    continue
                news_content = utils.read_json(news_content_path)
                news_title = news_content["title"] if "title" in news_content else ""
                news_description = news_content["meta_data"]["description"] \
                    if "description" in news_content["meta_data"] else ""
                if clean:
                    news_title = utils.clean_tweet_text(news_title)
                    news_description = utils.clean_tweet_text(news_description)
                if not os.path.isdir(tweet_dir):
                    continue
                for tweet_id in os.listdir(tweet_dir):
                    if i % self.info_every == 0:
                        print("Loaded {} Fake News Net tweets".format(str(i)))
                    tweet_path = os.path.join(tweet_dir, tweet_id)
                    tweet_content = utils.read_json(tweet_path)
                    tweet_text = utils.clean_tweet_text(tweet_content["text"]) if clean else tweet_content["text"].replace("\n", " ")
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
            tweet_text = utils.clean_tweet_text(tweet_json["text"]).strip().rstrip()
            if len(tweet_text) > 0:
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


class RumorEvalTwitter19(RumorEval17):
    def __init__(self, re_root, info_every=10):
        super(RumorEvalTwitter19, self).__init__(re_root, info_every)
        self.traindev = os.path.join(self.re_root, "rumoureval-2019-training-data")
        self.re_data = os.path.join(self.traindev, "twitter-english")
        self.headline_csv = os.path.join(self.re_root, "headlines_twitter.csv")

    def load_labels(self):
        re_train_label_path = os.path.join(self.traindev, "train-key.json")
        re_dev_label_path = os.path.join(self.traindev, "dev-key.json")
        all_train_labels = utils.read_json(re_train_label_path)
        task_a_train_labels = all_train_labels["subtaskaenglish"]
        all_dev_labels = utils.read_json(re_dev_label_path)
        task_a_dev_labels = all_dev_labels["subtaskaenglish"]
        raw_label_map = {**task_a_train_labels, **task_a_dev_labels}
        return {str(k): v for k, v in raw_label_map.items()}  # convert tweet id from int to string


class RumorEvalReddit19(RumorEvalTwitter19):
    def __init__(self, re_root, info_every=10):
        super(RumorEvalReddit19, self).__init__(re_root, info_every)
        self.re_data = os.path.join(self.traindev, "reddit-training-data")

    def load(self):
        label_map = self.load_labels()
        feature_set, label_set = [], []

        i = 0
        for discourse_id in os.listdir(self.re_data):
            if i % self.info_every == 0:
                print("Loaded {} rumor eval 19 reddit discourses".format(str(i)))
            discourse_id_path = os.path.join(self.re_data, discourse_id)
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

    @staticmethod
    def load_reddit_folder(tweet_folder_path):
        tweet_map = {}
        for tweet_path in os.listdir(tweet_folder_path):
            tweet_full_path = os.path.join(tweet_folder_path, tweet_path)
            tweet_json = utils.read_json(tweet_full_path)
            tweet_text = utils.clean_tweet_text(tweet_json["text"]).strip().rstrip()
            if len(tweet_text) > 0:
                tweet_map[str(tweet_json["id"])] = utils.clean_tweet_text(tweet_json["text"])
        return tweet_map


class FncLoader(BaseDatasetLoader):

    def __init__(self, fnc_root, info_every=1000, dataset=StanceDataset):
        super().__init__(info_every)
        self.fnc_root = fnc_root
        self.fnc_train_bodies = os.path.join(fnc_root, "train_bodies.csv")
        self.fnc_train_stances = os.path.join(fnc_root, "train_stances.csv")
        self.dataset = dataset

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
        return self.dataset(cleaned_feature_set, label_set)

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

        return self.dataset(split_feature_set, split_label_set)


class FncRelationLoader(FncLoader):
    def __init__(self, fnc_root, info_every=1000, dataset=RelationDataset):
        super(FncRelationLoader, self).__init__(fnc_root, info_every, dataset)

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
