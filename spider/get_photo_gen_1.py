import requests
import chardet  # 字符串和文件编码检测模块
from bs4 import BeautifulSoup  # 从HTML提取数据
import os
import urllib

'''
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = {'User-Agent': user_agent}

r = requests.get('http://ibm.neusoft.edu.cn/', params='page_id=6913')

if r.status_code == requests.codes.ok:
    print('status code is '+str(r.status_code))  # 响应码
    print(r.headers)  # 响应头
    print(r.headers.get('content-type'))  # 获取其中的某个字段
else:
    r.raise_for_status()

#print(r.content)
#print('-'*20+'\n'+'encoding is '+r.encoding+'\n'+'-'*20)
print(chardet.detect(r.content))
r.encoding = chardet.detect(r.content)['encoding']
#print(r.text)

soup = BeautifulSoup(r.text, 'html.parser', from_encoding='utf-8')
print(soup.prettify())
'''
def getSrc(url):
    headers = ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")
    opener = urllib.request.build_opener()
    opener.addheaders = [headers]
    res = requests.get(url)
    print('url is '+str(url))
    print('res is '+str(res))

    # 构建一个BeautifulSoup对象
    soup = BeautifulSoup(res.text, 'html.parser', from_encoding='utf-8')
    pic = soup.find_all('img')
    links = []
    for content in pic:
        print('content is '+str(content))
        s = content['src']
        if s is None:
            continue
        else:
            s2 = 'https://apod.nasa.gov/apod/'+s
            links.append(s2)
    print('-'*20+'\n'+"共有"+str(len(links))+"张图片")
    return links

def save(path, links):
    print('in save function, path is '+str(path))
    if(len(links)==0):
        print('没有符合条件的图片')
        return

    #判断本地是否有photo这个路径，没有的话创建一个
    if not os.path.exists(path):
        os.makedirs(path)
        print('debug-1')

    i = 0
    for link in links:
        print('in for loop '+str(link))
        i += 1
        filename = path+'photo-'+str(i)+'.jpg'
        print(filename)
        with open(filename, 'wb'):
            urllib.request.urlretrieve(link, filename)
        print('正在下载：'+str(link))
    print("图片下载完成")

def doCrew(url, path):
    links = getSrc(url)
    print('links are '+str(links))
    save(path, links)

if __name__ == "__main__":
    #url = 'https://www.runoob.com/'
    #url = 'https://tuijian.hao123.com/?type=rec'
    url = 'https://apod.nasa.gov/apod/astropix.html'

    path = 'D:\\Python\\get_photos\\'
    doCrew(url, path)

# test