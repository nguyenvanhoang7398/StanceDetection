import urllib3
from bs4 import BeautifulSoup

PAGE_NOT_FOUND = "Page not found"
MEDIA_NOT_FOUND = "Media not found"


class HeadlineCrawler(object):

    def __init__(self):
        self.http = urllib3.PoolManager()

    def crawl_url_title(self, url):
        soup = BeautifulSoup(self.http.request('GET', url).data)
        if soup.title is None or soup.title.string is None:
            return PAGE_NOT_FOUND, MEDIA_NOT_FOUND
        raw_title = soup.title.string
        last_hyphen_idx = raw_title.rfind("-")
        last_pipe_idx = raw_title.rfind("|")
        last_dash_idx = raw_title.rfind("—")
        split_idx = max([last_hyphen_idx, last_pipe_idx, last_dash_idx])
        if split_idx == -1:
            return raw_title, MEDIA_NOT_FOUND
        title, news_media = raw_title[:split_idx].strip(), raw_title[split_idx+1:]
        return title, news_media
