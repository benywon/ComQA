# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午2:47
 @FileName: __init__.py.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import json

from utils import *


def change_one(one_filename):
    data = []
    for line in tqdm(get_file_info(one_filename)):
        cc = json.loads(line)
        cc['nodes'] = cc['docs']
        del cc['docs']
        data.append(json.dumps(cc, ensure_ascii=False))
    write_lst_to_file(data, one_filename)


if __name__ == '__main__':
    change_one('../data/comqa/train.json')
    change_one('../data/comqa/dev.json')
    change_one('../data/comqa/test.json')
