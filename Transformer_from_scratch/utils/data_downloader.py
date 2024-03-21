import os
import requests


def download(url, dir, name=None):
    os.makedirs(dir, exist_ok=True)
    if name is None:
        name = url.split('/')[-1]
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Install {name} ...')
        open(path, 'wb').write(requests.get(url).content)
        print('Install successfully.')


def download_data():
    data_dir = '../datasets/translation_corpus'
    urls = [
        'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/master/corpora/cn.txt',
        'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/master/corpora/en.txt',
        'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/master/preprocessed/cn.txt.vocab.tsv',
        'https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/master/preprocessed/en.txt.vocab.tsv'
    ]
    for url in urls:
        download(url, data_dir)


if __name__ == '__main__':
    download_data()
