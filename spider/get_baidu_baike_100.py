#coding:utf-8
import requests
import re
import urllib.parse
from bs4 import BeautifulSoup
import codecs

class UrlManager(object):
    def __init__(self):
        self.new_urls = set()  # 未爬取URL集合
        self.old_urls = set()  # 已爬取URL集合

    def has_new_url(self):
        # 判断是否有未爬取的URL
        return self.new_url_size()!=0

    def get_new_url(self):
        # 获取一个未爬取的URL
        new_url = self.new_urls.pop()
        self.old_urls.add(new_url)
        return new_url

    def add_new_url(self, url):
        # 将新的URL添加到未爬取的URL集合中
        if url is None:
            return
        if url not in self.new_urls and url not in self.old_urls:
            self.new_urls.add(url)

    def add_new_urls(self, urls):
        # 将新的URL添加到未爬取的URL集合中
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)

    def new_url_size(self):
        # 获取未爬取URL集合的s大小
        return len(self.new_urls)

    def old_url_size(self):
        # 获取已经爬取URL集合的大小
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
        # 用于解析网页内容抽取URL和数据
        # param page_url: 下载页面的URL
        # param html_cont: 下载的网页内容
        if page_url is None or html_cont is None:
            return
        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_data(page_url, soup)
        return new_urls, new_data

    def _get_new_urls(self, page_url, soup):
        # 抽取新的URL集合
        # param page_url: 下载页面的URL
        # param soup:soup
        new_urls = set()
        #抽取符合要求的a标签
        links = soup.find_all('a', \
                              href=re.compile(r'/item/.*'))  # 一个网站一个样
        for link in links:
            #提取href属性
            new_url = link['href']
            #拼接成完整网址
            new_full_url = urllib.parse.urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self, page_url, soup):
        # 抽取有效数据
        # param page_url:下载页面的URL
        # param soup:
        data = {}
        data['url'] = page_url
        title = soup.find('dd', class_='lemmaWgt-lemmaTitle-title').find('h1')
        data['title'] = title.get_text()
        summary = soup.find('div', class_='lemma-summary')
        # 获取到tag中包含的所有文版内容包括子孙tag中的内容,并将结果作为Unicode字符串返回
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
        # 添加入口URL
        self.manager.add_new_url(root_url)
        # 判断url管理器中是否有新的url，同时判断抓取了多少个url
        while self.manager.has_new_url() and self.manager.old_url_size() < 100:
            try:
                # 从URL管理器获取新的url
                new_url = self.manager.get_new_url()
                # HTML下载器下载网页
                html = self.downloader.download(new_url)
                # HTML解析器抽取网页数据
                new_urls,data = self.parser.parser(new_url,html)
                # 将抽取到url添加到URL管理器中
                self.manager.add_new_urls(new_urls)
                # 数据存储器储存文件
                self.output.store_data(data)
                print("已经抓取%s个链接"%self.manager.old_url_size())
            except Exception as e:
                print("crawl failed")
            # 数据存储器将文件输出成指定格式
        self.output.output_html()

if __name__=="__main__":
    spider_man = SpiderMan()
    #spider_man.crawl("http://baike.baidu.com/view/284853.htm")
    spider_man.crawl("https://baike.baidu.com/item/%E5%A4%A7%E8%BF%9E%E4%B8%9C%E8%BD%AF%E4%BF%A1%E6%81%AF%E5%AD%A6%E9%99%A2/4625571")
