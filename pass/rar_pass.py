from unrar import rarfile
import os
import itertools as its
import time
from concurrent.futures import ThreadPoolExecutor

def get_pwd(file_path, output_path, pwd):
    '''
    �ж������Ƿ���ȷ
    :param file_path: ��Ҫ�ƽ���ļ�·����������Ե����ļ������ƽ�
    :param output_path: ��ѹ����ļ�·��
    :param pwd: ���������
    :return:
    '''
    # ���뱻��ѹ���ļ�·�������ɴ���ѹ�ļ�����
    file = rarfile.RarFile(file_path)
    # �����ѹ����ļ�·��
    out_put_file_path = 'rar/{}'.format(file.namelist()[0])
    file.extractall(path=output_path, pwd=pwd)
    try:
        # ��������ļ�����ѹ�����Ƴ����ļ�
        os.remove(out_put_file_path)
        # ˵����ǰ������Ч������֪
        print('Find password is "{}"'.format(pwd))
        return True
    except Exception as e:
        # ���벻��ȷ
        print('"{}" is nor correct password!'.format(pwd))
        # print(e)
        return False

def get_password(min_digits, max_digits, words):
    """
    ����������
    :param min_digits: ������С����
    :param max_digits: ������󳤶�
    :param words: ��������漰���ַ�
    :return: ����������
    """
    while min_digits <= max_digits:
        pwds = its.product(words, repeat=min_digits)
        for pwd in pwds:
            yield ''.join(pwd)
        min_digits += 1

file_path = 'C:\\Users\\Admin\\Desktop\\goal.rar'
output_path = 'C:\\Users\\Admin\\Desktop\\rar_pass'
print('debug-1')

# ���뷶Χ
words = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # �漰����������Ĳ���

pwds = get_password(1, 8, words)
# ��ʼ��������
start = time.time()
print('start -- ysd')
while True:
    pwd = next(pwds)
    if get_pwd(file_path, output_path, pwd=pwd):
        break
end = time.time()
print('�����ʱ{}'.format(end - start))

