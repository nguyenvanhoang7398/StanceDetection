from difflib import SequenceMatcher
import preprocessor as p
import re
import string
import wordninja
import utils.io as utils_io


COMMON_ENGLISH_WORDS = utils_io.load_text_as_list("datasets/common_20k.txt")


def strip_urls(text):
    url_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()“”~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url[0], ', ')
    return text


def strip_tags(text):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, ' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def is_number(s):
    return s.replace('.', '', 1).isdigit()


def simple_clean(text):
    return str(text).lower().replace("\n", " ").replace("\t", " ").strip().rstrip()


def clean_tweet_text(tweet_text): 
    tweet_text = tweet_text.replace("’", "'").replace("…", "...")
    tweet_parser = p.parse(tweet_text)
    cleaned_tweet = tweet_text
    hash_tags = tweet_parser.hashtags
    if hash_tags is not None:
        for hash_tag in hash_tags:
            cleaned_tweet = cleaned_tweet.replace(hash_tag.match, " ".join(wordninja.split(hash_tag.match[1:])))
    tweet_urls = tweet_parser.urls
    if tweet_urls is not None:
        for url_link in tweet_urls:
            cleaned_tweet = cleaned_tweet.replace(url_link.match, " url$$ ")
    tweet_emojis = tweet_parser.emojis
    if tweet_emojis is not None:
        for emoji in tweet_emojis:
            cleaned_tweet = cleaned_tweet.replace(emoji.match, " emoji$$ ")
    cleaned_tweet = cleaned_tweet.split("via")[0].split("|")[0].split(" - ")[0].split(" – ")[0]
    cleaned_tweet_tokens = []
    for word_token in cleaned_tweet.split(" "):
        word_token = word_token.strip().rstrip()
        if word_token.endswith("$$") or word_token in COMMON_ENGLISH_WORDS:
            cleaned_tweet_tokens.append(word_token)
        elif len(word_token) > 0:
            split_tokens = [w for w in wordninja.split(word_token) if w not in string.punctuation]
            cleaned_tweet_tokens += [token for token in split_tokens if not is_number(token)]

    cleaned_tweet = " ".join(cleaned_tweet_tokens)
    return cleaned_tweet


def longest_common_substring(m, n):
    seq_matcher = SequenceMatcher(None, m, n)
    match = seq_matcher.find_longest_match(0, len(m), 0, len(n))
    if match.size != 0:
        return m[match.a: match.a + match.size]
    return ""


def remove_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def clean_stance_target(target, source):
    while True:
        common_substring = longest_common_substring(target, source)
        if len(common_substring) > 0 and len(common_substring) >= len(target) / 3.0:
            target = target.replace(common_substring, "")
        else:
            break
    return remove_punctuations(target)

