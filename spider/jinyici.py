__author__ = 'jeff'

import requests
from bs4 import BeautifulSoup
import json
import re
from tqdm import tqdm


class JinYiCi(object):

    headers = {'charset': 'utf-8'}

    def __init__(self):
        self.url_dict = {}
        self.error_list = []
        self.tongyi_err = []
        self.tongyi_dict = {}
        self.fanyi_dict = {}
        self.similar_dict = {}

    def pares_word(self, soup):
        """"""
        for li in soup.find_all('li'):
            word = li.get_text()
            word_url = li.a['href']
            self.url_dict[word] = word_url

    def get_word_list(self):
        """"""
        start_py = list('abcdefghjklmnopqrstwxyz')
        temp_url = "https://www.diyifanwen.com/jinyici/jinyici-%s/"
        for i in tqdm(start_py):
            url = temp_url % i.upper()
            r = requests.request('get', url)
            r.encoding = 'gbk'
            if r.status_code == 200:
                soup = BeautifulSoup(r.text)
                self.pares_word(soup)
                k = 1
                while True:
                    next_url = None
                    k += 1
                    page_list = soup.select('div[id=CutPage]')

                    if not page_list:
                        self.error_list.append(i)
                        break
                    page_list = page_list[0].find_all()

                    for page in page_list:
                        if page.get_text() != '下一页':
                            continue
                        next_url = page.get('href')
                        if not next_url:
                            break

                    if not next_url:
                        break
                    r = requests.get('https:'+next_url)
                    if r.status_code != 200:
                        self.error_list.append(i+str(k))
                    r.encoding = 'gbk'
                    new_s = BeautifulSoup(r.text)
                    self.pares_word(new_s)
                    soup = new_s
                    print('===%s==%s' % (i, str(k)))
                    if k > 1000:
                        print('======翻页错误， 无限循环')
                        break

            else:
                self.error_list.append(i)

        self.save('url_dict', self.url_dict)
        self.save('err_list', self.error_list)

    def load_url_dict(self):
        with open('data/url_dict.json', 'r', encoding='utf-8') as f:
            k = f.readlines()
            k = ''.join(k)
            if k:
                self.url_dict = json.loads(k)

    def save(self, name, data):
        """"""
        with open('data/%s.json' % name, 'w', encoding='utf-8') as f:
            j = json.dumps(data, ensure_ascii=False)
            f.write(j)
            f.flush()

    def load_tongyi(self):
        with open('data/tongyi.json', 'r', encoding='utf-8') as f:
            k = f.readlines()
            k = ''.join(k)
            if k:
                self.tongyi_dict = json.loads(k)

    def crawl(self):
        """爬取"""
        self.load_url_dict()
        self.load_tongyi()
        for word, url in tqdm(self.url_dict.items()):
            if self.tongyi_dict.get(word, None) is not None:
                continue
            new_url = 'https:' + url
            r = requests.get(new_url)
            r.encoding = 'gbk'
            soup = BeautifulSoup(r.text)
            tongyi = soup.select('div[id|=intro]')[0].get_text()
            tongyi_list = re.split('[【】]', tongyi)
            if len(tongyi_list) == 5:
                self.tongyi_dict[word] = tongyi_list[2]
                self.fanyi_dict[word] = tongyi_list[4]
            else:
                self.tongyi_err.append(word)
            s_word = []
            for li in soup.select('div[id|=RelateJYC]')[0].find_all('li'):
                if li.get('class') is not None:
                    continue
                similary_word = li.get_text()
                s_word.append(similary_word)
            self.similar_dict[word] = s_word
            # break
        self.save('tongyi', self.tongyi_dict)
        self.save('fanyi', self.fanyi_dict)
        self.save('similary', self.similar_dict)
        self.save('tongyi_err', self.tongyi_err)


if __name__ == '__main__':
    JinYiCi().crawl()
    # JinYiCi().get_word_list()


