import os
import codecs
import random
import numpy as np
import regex
import requests
import config


# words whose occurred less than min_cnt are encoded as <UNK>
min_cnt = 0
# maximum number of words in a sentence
max_seq_len = config.max_seq_len

"""
Data from https://github.com/P3n9W31/transformer-pytorch
"""
source_train = './datasets/translation_corpus/cn.txt'
target_train = './datasets/translation_corpus/en.txt'
source_test = './datasets/translation_corpus/en.test.txt'
target_test = './datasets/translation_corpus/en.test.txt'


def load_vocab(language):
    assert language in ['cn', 'en']
    vocab = [
        line.split()[0] for line in codecs.open(
            './datasets/translation_corpus/{}.txt.vocab.tsv'.format(language), 'r', 'utf-8'
        ).read().splitlines()
        if int(line.split()[1]) >= min_cnt
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_cn_vocab():
    word2idx, idx2word = load_vocab('cn')
    return word2idx, idx2word

def load_en_vocab():
    word2idx, idx2word = load_vocab('en')
    return word2idx, idx2word

def create_data(src_sentences, trg_sentences):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    # index
    x_list, y_list, Sources, Targets = [], [], [], []
    for src_sentence, trg_sentence in zip(src_sentences, trg_sentences):
        x = [
            cn2idx.get(word, 1)
            for word in ('<S> ' + src_sentence + ' </S>').split()
        ]  # 1: Out of Vocabulary, </S>: End of Text

        y = [
            en2idx.get(word, 1)
            for word in ('<S> ' + trg_sentence + ' </S>').split()
        ]

        if max(len(x), len(y)) <= max_seq_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(src_sentence)
            Targets.append(trg_sentence)

    # pad
    X = np.zeros([len(x_list), max_seq_len], np.int32)
    Y = np.zeros([len(y_list), max_seq_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, max_seq_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, max_seq_len - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets

def load_data(data_type):
    if data_type == 'train':
        source, target = source_train, target_train
    elif data_type == 'test':
        source, target = source_test, target_test

    assert data_type in ['train', 'test']

    cn_sentences = [
        regex.sub(r"[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(source, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]

    en_sentences = [
        regex.sub(r"[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(target, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]

    X, Y, Sources, Targets = create_data(cn_sentences, en_sentences)
    return X, Y, Sources, Targets

def load_train_data():
    X, Y, _, _ = load_data('train')
    return X, Y

def load_test_data():
    X, Y, _, _ = load_data('test')
    return X, Y

def get_batch_indices(total_length, batch_size):
    assert (batch_size <= total_length), (
        'Batch size is large than total datasets length.' 
        'Check your datasets or change batch size.'
    )

    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index:current_index + batch_size], current_index

def idx_to_sentence(arr, vocab, insert_space=False):
    res = ''
    first_word = True
    for id in arr:
        word = vocab[id.item()]

        if insert_space and not first_word:
            res += ' '
        first_word = False

        res += word

    return res
