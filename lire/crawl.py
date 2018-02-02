import requests
from urllib.parse import urlencode

# https://www.vcg.com/api/common/searchImage?phrase=%E6%BB%91%E9%9B%AA&rand=LLLO18_79b3c280d887ae638450b50a83ebbf01&page=2&keyword=%E6%BB%91%E9%9B%AA

query = {
    'phrase': '滑雪',
    'rand': 'LLLO18_79b3c280d887ae638450b50a83ebbf01',
    'page': '1',
    'keyword': '滑雪'
}

base_url = 'https://www.vcg.com/api/common/searchImage'

headers = {
    'user-agent': 'Baiduspider'
}


def write_result(content):
    with open('result.txt', 'ab') as f:
        f.write(content)
        f.write(b'\n')

def crawl(page):
    q = query.copy()
    q['page'] = page

    url = '{}?{}'.format(base_url, urlencode(q))
    rsp = requests.get(url, headers=headers)
    write_result(rsp.content)

for page in range(1, 152):
    print('crawl page: ', page)
    crawl(page)
    print('crawled page: ', page)
