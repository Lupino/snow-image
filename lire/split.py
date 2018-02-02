START_DOC = '<doc>'
END_DOC = '</doc>'

BATCH_SIZE = 3000

def write_batch(batch, count):
    batch = '\n'.join(batch)
    with open('batch_{}.xml'.format(count), 'w') as fo:
        fo.write('<add>\n')
        fo.write(batch)
        fo.write('</add>\n')


with open('output-preprocess.xml', 'r') as f:
    data = f.read()

    count = 0
    batch = []
    ident = 0
    while True:
        start = data.find(START_DOC, ident)
        if start == -1:
            break
        end = data.find(END_DOC, ident)
        if end == -1:
            break

        end = end + len(END_DOC)
        ident = end + 1

        batch.append(data[start:end])

        if len(batch) >= BATCH_SIZE:
            write_batch(batch, count)
            batch = []
            count += 1

    if len(batch) > 0:
        write_batch(batch, count)
