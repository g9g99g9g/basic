import requests
import chardet  # �ַ������ļ�������ģ��
from bs4 import BeautifulSoup  # ��HTML��ȡ����
import os
import urllib

'''
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = {'User-Agent': user_agent}

r = requests.get('http://ibm.neusoft.edu.cn/', params='page_id=6913')

if r.status_code == requests.codes.ok:
    print('status code is '+str(r.status_code))  # ��Ӧ��
    print(r.headers)  # ��Ӧͷ
    print(r.headers.get('content-type'))  # ��ȡ���е�ĳ���ֶ�
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

    # ����һ��BeautifulSoup����
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
    print('-'*20+'\n'+"����"+str(len(links))+"��ͼƬ")
    return links

def save(path, links):
    print('in save function, path is '+str(path))
    if(len(links)==0):
        print('û�з���������ͼƬ')
        return

    #�жϱ����Ƿ���photo���·����û�еĻ�����һ��
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
        print('�������أ�'+str(link))
    print("ͼƬ�������")

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