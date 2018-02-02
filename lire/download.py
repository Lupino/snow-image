import json
import requests

def image_url():
    with open('result.txt', 'r') as f:
        for line in f:
            data = json.loads(line)
            for dt in data['list']:
                yield {
                    'id': dt['id'],
                    'img_url': 'https:' + dt['equalh_url']
                }

headers = {
    'user-agent': 'Baiduspider'
}

for img in image_url():
    print('download', img['img_url'])
    rsp = requests.get(img['img_url'], headers=headers)
    with open('images/{}.jpg'.format(img['id']), 'wb') as f:
        f.write(rsp.content)
