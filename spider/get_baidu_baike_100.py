#coding:utf-8
import requests
import re
import urllib.parse
from bs4 import BeautifulSoup
import codecs

class UrlManager(object):
    def __init__(self):
        self.new_urls = set()  # δ��ȡURL����
        self.old_urls = set()  # ����ȡURL����

    def has_new_url(self):
        # �ж��Ƿ���δ��ȡ��URL
        return self.new_url_size()!=0

    def get_new_url(self):
        # ��ȡһ��δ��ȡ��URL
        new_url = self.new_urls.pop()
        self.old_urls.add(new_url)
        return new_url

    def add_new_url(self, url):
        # ���µ�URL��ӵ�δ��ȡ��URL������
        if url is None:
            return
        if url not in self.new_urls and url not in self.old_urls:
            self.new_urls.add(url)

    def add_new_urls(self, urls):
        # ���µ�URL��ӵ�δ��ȡ��URL������
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)

    def new_url_size(self):
        # ��ȡδ��ȡURL���ϵ�s��С
        return len(self.new_urls)

    def old_url_size(self):
        # ��ȡ�Ѿ���ȡURL���ϵĴ�С
        return len(self.old_urls)

class HtmlDownloader(object):
    def download(self, url):
        if url is None:
            return None
        user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        headers = {'User-Agent':user_agent}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            r.encoding = 'utf-8'
            return r.text
        return None

class HtmlParser(object):
    def parser(self, page_url, html_cont):
        # ���ڽ�����ҳ���ݳ�ȡURL������
        # param page_url: ����ҳ���URL
        # param html_cont: ���ص���ҳ����
        if page_url is None or html_cont is None:
            return
        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_data(page_url, soup)
        return new_urls, new_data

    def _get_new_urls(self, page_url, soup):
        # ��ȡ�µ�URL����
        # param page_url: ����ҳ���URL
        # param soup:soup
        new_urls = set()
        #��ȡ����Ҫ���a��ǩ
        links = soup.find_all('a', \
                              href=re.compile(r'/item/.*'))  # һ����վһ����
        for link in links:
            #��ȡhref����
            new_url = link['href']
            #ƴ�ӳ�������ַ
            new_full_url = urllib.parse.urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self, page_url, soup):
        # ��ȡ��Ч����
        # param page_url:����ҳ���URL
        # param soup:
        data = {}
        data['url'] = page_url
        title = soup.find('dd', class_='lemmaWgt-lemmaTitle-title').find('h1')
        data['title'] = title.get_text()
        summary = soup.find('div', class_='lemma-summary')
        # ��ȡ��tag�а����������İ����ݰ�������tag�е�����,���������ΪUnicode�ַ�������
        data['summary'] = summary.get_text()
        return data

class DataOutput(object):
    def __init__(self):
        self.datas = []
    def store_data(self, data):
        if data is None:
            return
        self.datas.append(data)
    def output_html(self):
        fout = codecs.open('D:\\Python\\get_photos\\baike.html', \
                           'w', encoding='utf-8')
        fout.write("<html>")
        fout.write("<head><meta charset='utf-8'/></head>")
        fout.write("<body>")
        fout.write("<table>")
        for data in self.datas:
            fout.write("<tr>")
            fout.write("<td>%s</td>"%data['url'])
            fout.write("<td>%s</td>"%data['title'])
            fout.write("<td>%s</td>"%data['summary'])
            fout.write("</tr>")
        fout.write("</table>")
        fout.write("</body>")
        fout.write("</html>")
        fout.close()

class SpiderMan(object):
    def __init__(self):
        self.manager = UrlManager()
        self.downloader = HtmlDownloader()
        self.parser = HtmlParser()
        self.output = DataOutput()
    def crawl(self, root_url):
        # ������URL
        self.manager.add_new_url(root_url)
        # �ж�url���������Ƿ����µ�url��ͬʱ�ж�ץȡ�˶��ٸ�url
        while self.manager.has_new_url() and self.manager.old_url_size() < 100:
            try:
                # ��URL��������ȡ�µ�url
                new_url = self.manager.get_new_url()
                # HTML������������ҳ
                html = self.downloader.download(new_url)
                # HTML��������ȡ��ҳ����
                new_urls,data = self.parser.parser(new_url,html)
                # ����ȡ��url��ӵ�URL��������
                self.manager.add_new_urls(new_urls)
                # ���ݴ洢�������ļ�
                self.output.store_data(data)
                print("�Ѿ�ץȡ%s������"%self.manager.old_url_size())
            except Exception as e:
                print("crawl failed")
            # ���ݴ洢�����ļ������ָ����ʽ
        self.output.output_html()

if __name__=="__main__":
    spider_man = SpiderMan()
    #spider_man.crawl("http://baike.baidu.com/view/284853.htm")
    spider_man.crawl("https://baike.baidu.com/item/%E5%A4%A7%E8%BF%9E%E4%B8%9C%E8%BD%AF%E4%BF%A1%E6%81%AF%E5%AD%A6%E9%99%A2/4625571")
