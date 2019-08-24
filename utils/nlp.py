from difflib import SequenceMatcher
import preprocessor as p
import re
import string


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


def clean_tweet_text(tweet_text):
    tweet_text = tweet_text.replace("’", "'").replace("…", "...")
    return remove_punctuations(p.clean(tweet_text))


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

