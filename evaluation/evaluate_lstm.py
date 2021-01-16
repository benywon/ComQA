# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/16 上午10:54
 @FileName: evaluate_lstm.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import sys

sys.path.append('..')
import torch
from apex import amp
from modules.LSTM import LSTM
from utils import *

test_data = load_file(sys.argv[1])
print('test size is {}'.format(len(test_data)))

vocab_size = 50000
n_embedding = 128
n_hidden = 512
n_layer = 2
batch_size = 16
doc_max_length_size = 1024
model = LSTM(vocab_size, n_embedding, n_hidden, n_layer)

filename = sys.argv[2]
state_dict = load_file(filename)
for name, para in model.named_parameters():
    if name not in state_dict:
        print('{} not load'.format(name))
        continue
    para.data = torch.FloatTensor(state_dict[name])

model.cuda()
[model] = amp.initialize([model], opt_level="O2", verbosity=0)
model.eval()

test_data = sorted(test_data, key=lambda x: len(x[0]))


def get_train_data(batch, max_len=doc_max_length_size):
    batch, _ = padding(batch, pads=0, max_len=max_len)
    seq = batch.flatten()
    real_end_pos = np.where(seq == -1)[0]
    np.put(seq, real_end_pos, vocab_size)
    all_end_pos_seq = np.where(seq == vocab_size)[0]
    label = np.zeros(shape=len(all_end_pos_seq), dtype='float32')
    for i, j in enumerate(all_end_pos_seq):
        if j in real_end_pos:
            label[i] = 1
    batch = seq.reshape(batch.shape)
    return batch, label


total = len(test_data)
results = []
with torch.no_grad():
    for i in tqdm(range(0, total, batch_size)):
        sample = test_data[i:i + batch_size]
        context_raw = [x[0] for x in sample]
        paras = [x[1] for x in sample]
        batch, label = get_train_data(context_raw, doc_max_length_size)
        batch = torch.LongTensor(batch)
        mask_idx = torch.eq(batch, vocab_size)
        answer_logits = model([batch.cuda(), None])
        end_num = mask_idx.sum(1).data.numpy().tolist()
        answer_logits = answer_logits.cpu().data.numpy().tolist()
        start = 0
        for one_sent_end_num, para in zip(end_num, paras):
            pred = answer_logits[start:start + one_sent_end_num]
            results.append([pred, para])
            start += one_sent_end_num
    threshold = 0.5
    precision, recall, f1, macro_f1, accuracy = evaluate_comqa(results, )
    print('Test results:\nthreshold:{}\nprecision:{}\nrecall:{}\nf1:{}\nmacro_f1:{}\naccuracy:{}'.format(
        threshold, precision,
        recall, f1,
        macro_f1, accuracy))
