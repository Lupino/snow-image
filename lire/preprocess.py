import re
import json

# <doc><field name="id">/Users/lmj/yuntan/experiment/snow/images/1134820197.jpg</field>
# <field name="title">/Users/lmj/yuntan/experiment/snow/images/1134820197.jpg</field>
# <field name="jc_hi">BQH+AfMDDgj9AQLwBATqBAb8AQHwAgUB6wIC+wLqAgLwAgL8</field>

re_id = re.compile('<field name="(id|title)">[^0-9]+(\d+).jpg</field>')

def group_images():
    retval = {}
    with open('result.txt', 'r') as f:
        for line in f:
            data = json.loads(line)
            for dt in data['list']:
                retval[dt['id']] = {
                    'id': dt['id'],
                    'title': dt['title'],
                    'imgurl': dt['equalh_url']
                }

    return retval

all_images = group_images()

def dashrepl(m):
    k = m.group(1)
    id = int(m.group(2))
    img = all_images[id]
    fields = []
    # fields.append('<field name="{}">{}</field>'.format(k, id))
    if k == 'id':
        fields.append('<field name="{}">https://www.vcg.com/creative/{}</field>'.format(k, id))
        fields.append('<field name="{}">https:{}</field>'.format('imgurl', img['imgurl']))

    if k == 'title':
        fields.append('<field name="{}">{}</field>'.format(k, img['title'].replace('&', '')))

    return '\n'.join(fields)

with open('output.xml', 'r') as fi:
    data = fi.read()
    data = re_id.sub(dashrepl, data)
    with open('output-preprocess.xml', 'w') as fo:
        fo.write(data)

