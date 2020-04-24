__author__ = 'jeff'

# encoding:utf-8

import requests
import base64

'''
手写文字识别
'''


def get_access_token():
    """"""
    ak = 'agr6OSpx7ou4UDWcFbElWTjL'
    sk = '6iphA0G9xAZfznF18etMi73oakHwYNTC'
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(ak, sk)
    r = requests.get(host)
    if r:
        print(r.json())
        return r.json()['access_token']
    # return r


def hand_write(token, file_name):
    """"""
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
    # 二进制方式打开图片文件
    with open(file_name, 'rb') as f:
        img = base64.b64encode(f.read())

        params = {"image": img}
        request_url = request_url + "?access_token=" + token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            print(response.json())


def general_basic(token, file_name):
    """"""
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    # 二进制方式打开图片文件
    with open(file_name, 'rb') as f:
        img = base64.b64encode(f.read())
        params = {"image": img}
        request_url = request_url + "?access_token=" + token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            print(response.json())


if __name__ == '__main__':
    f = 'E:\数据\信件图片\微信图片_20200416134938.jpg'
    access_token = get_access_token()
    # hand_write(access_token, f)
    general_basic(access_token, f)

