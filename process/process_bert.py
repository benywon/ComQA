# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午4:29
 @FileName: process_bert.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import argparse
import json
import sys
sys.path.append('..')
from utils import *
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('bpe.50000.new.model')
vocab_size = sp.vocab_size()

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train',
    type=str,
    required=True,
    help=
    'the training file path'
)
parser.add_argument(
    '--dev',
    type=str,
    required=True,
    help=
    'the dev file path'
)
parser.add_argument(
    '--test',
    type=str,
    required=True,
    help=
    'the test file path'
)
args = parser.parse_args()


def process_one_line(line):
    sample = json.loads(line)
    url = sample['url']
    question = clean(sample['question'])
    paras = sample['nodes']
    seq_ids = sp.EncodeAsIds(clean(question)) + [1]  # 1 is the question segment indicator
    for para in paras:
        text = para[0]
        label = para[1]
        p_lst = sp.EncodeAsIds(clean(text))
        p_lst.append(-1 if label == 1 else vocab_size)
        seq_ids.extend(p_lst)
    return [seq_ids, paras, url]


def process_one(filename):
    features = multi_process(process_one_line, get_file_info(filename))
    return features


def process_comqa_to_features():
    train_path = '../data/comqa/train.json'
    data = process_one(train_path)
    print('train data size is {}'.format(len(data)))
    dump_file(data, args.train)

    dev_path = '../data/comqa/dev.json'
    data = process_one(dev_path)
    print('dev data size is {}'.format(len(data)))
    dump_file(data, args.dev)

    test_path = '../data/comqa/test.json'
    data = process_one(test_path)
    print('test data size is {}'.format(len(data)))
    dump_file(data, args.test)


if __name__ == '__main__':
    process_comqa_to_features()
