# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/17 下午2:42
 @FileName: utils.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import itertools
import multiprocessing
import pickle
import re

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(10245)


def get_file_charset(filename):
    import chardet
    rawdata = open(filename, 'rb').read(1000)
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return charenc


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def write_lst_to_file(lst, filename, encoding='utf-8'):
    output = '\n'.join(lst)
    with open(filename, 'w', encoding=encoding, errors='ignore') as f:
        f.write(output)


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_model_parameters(model):
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            tmp = 1
            for a in parameter.size():
                tmp *= a
            total += tmp
    return total


def remove_duplciate_lst(lst):
    lst.sort()
    return list(k for k, _ in itertools.groupby(lst))


def padding(sequence, pads=0, max_len=None, dtype='int32', return_matrix_for_size=False):
    # we should judge the rank
    if True or isinstance(sequence[0], list):
        v_length = [len(x) for x in sequence]  # every sequence length
        seq_max_len = max(v_length)
        if (max_len is None) or (max_len > seq_max_len):
            max_len = seq_max_len
        v_length = list(map(lambda z: z if z <= max_len else max_len, v_length))
        x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
        for idx, s in enumerate(sequence):
            trunc = s[:max_len]
            x[idx, :len(trunc)] = trunc
        if return_matrix_for_size:
            v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                     dtype=dtype)
            return x, v_matrix
        return x, np.asarray(v_length, dtype='int32')
    else:
        seq_len = len(sequence)
        if max_len is None:
            max_len = seq_len
        v_vector = sequence + [0] * (max_len - seq_len)
        padded_vector = np.asarray(v_vector, dtype=dtype)
        v_index = [1] * seq_len + [0] * (max_len - seq_len)
        padded_index = np.asanyarray(v_index, dtype=dtype)
        return padded_vector, padded_index


def add2count(value, map):
    if value not in map:
        map[value] = 0
    map[value] += 1


import os


def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L


def clean(txt):
    txt = DBC2SBC(txt)
    txt = txt.lower()
    txt = re.sub('(\s*)?(<.*?>)?', '', txt)
    return txt


def multi_process(func, lst, num_cores=multiprocessing.cpu_count(), backend='multiprocessing'):
    workers = Parallel(n_jobs=num_cores, backend=backend)
    output = workers(delayed(func)(one) for one in tqdm(lst))
    # output = workers(delayed(func)(one) for one in lst)
    return output


def get_file_info(filename):
    with open(filename, encoding=get_file_charset(filename), errors='ignore') as f:
        for line in f:
            yield line


def evaluate_comqa(results, threshold=0.5):
    precision = []
    recall = []
    f1 = []
    accuracy = []
    for one in results:
        [pred, paras] = one
        sample_a = 1.0e-9
        sample_b = 1.0e-9
        sample_c = 1.0e-9
        num = 0
        if len(pred) < len(paras):
            pred.extend([0.0] * len(paras))
        for p, para in zip(pred, paras):
            r = para[1]
            num += 1
            if p > threshold:
                sample_a += 1
            if r == 1:
                sample_b += 1
            if p > threshold and r == 1:
                sample_c += 1
        sample_precision = sample_c / sample_a
        sample_recall = sample_c / sample_b
        if sample_precision >= 0.999 and sample_recall >= 0.999:
            acc = 1
        else:
            acc = 0
        sample_f1 = 2 * sample_precision * sample_recall / (sample_recall + sample_precision)
        precision.append(sample_precision)
        recall.append(sample_recall)
        f1.append(sample_f1)
        accuracy.append(acc)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = np.mean(f1)
    accuracy = np.mean(accuracy)
    macro_f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, macro_f1, accuracy
