# -*- coding: utf-8 -*-

import requests
import uuid
from lxml import html
from settings import *

def get_photo_from_url(client, url):
    # 爬取图片
    r = client.get(url, stream=True)
    if r.status_code == 200:
        with open("test_image/%s.jpg" % (uuid.uuid4().hex,), 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    else:
        pass

def main():

    # html1 = requests.get(url)
    # html1 = html.fromstring(html1.text)
    # result = html1.xpath('//*[@id="login-form"]/input/@value')[0]

    client = requests.session()
    client.get(URL)  # sets cookie
    csrftoken = client.cookies['csrftoken']
    payload = {
        'csrfmiddlewaretoken': csrftoken,
        'username': USERNAME,
        'password': PASSWORD,
        'next': '/admin/'
    }

    r_login = client.post(URL, data=payload, headers=dict(Referer=URL))

    r_data_list = client.get(URL_ROOT + '/admin/exam/photo/')
    html_data_list = html.fromstring(r_data_list.text)
    results = html_data_list.xpath('//*[@id="result_list"]/tbody//tr/th/a/@href')
    for result in results:
        url = URL_ROOT + result
        r_data = client.get(url)
        html_data = html.fromstring(r_data.text)
        if html_data.xpath('//*[@id="photo_form"]/div/fieldset/div[3]/div/p/a/@href'):
            result_url = 'http' + html_data.xpath('//*[@id="photo_form"]/div/fieldset/div[3]/div/p/a/@href')[0][5:]
            get_photo_from_url(client, result_url)

if __name__ == '__main__':
    main()
