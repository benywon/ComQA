# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午7:20
 @FileName: process_official_bert.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse
import json
import sys

sys.path.append('..')
from utils import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

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
    seq_ids = tokenizer.encode(clean(question), truncation=True, max_length=100, add_special_tokens=False) + [
        tokenizer.sep_token_id]
    for para in paras:
        text = para[0]
        label = para[1]
        p_lst = tokenizer.encode(clean(text), truncation=True, max_length=200, add_special_tokens=False)
        p_lst.append(-1 if label == 1 else tokenizer.cls_token_id)
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
