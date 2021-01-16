# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午9:18
 @FileName: official_bert.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import sys
import time

import apex
import torch
import torch.distributed as dist
from apex import amp
from transformers import AutoTokenizer

sys.path.append('..')
from modules.OfficialBERT import OfficialBERT
from train.parser import get_argument_parser

from utils import *

np.random.seed(1000)
torch.manual_seed(1024)
torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
args = get_argument_parser()
print(args.local_rank, dist.get_rank(), dist.get_world_size())
torch.cuda.set_device(args.local_rank)

batch_size = 8
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
max_learning_rate = 5e-5
doc_max_length_size = 512

train_data = load_file(args.train_file_path)
dev_data = load_file(args.dev_file_path)
dev_data = sorted(dev_data, key=lambda x: len(x[0]))
remove_data_size = len(dev_data) % dist.get_world_size()
thread_dev_data = [dev_data[x + args.local_rank] for x in
                   range(0, len(dev_data) - remove_data_size, dist.get_world_size())]

print('train data size is {} test size {}'.format(len(train_data), len(dev_data)))
model = OfficialBERT(indicator=tokenizer.cls_token_id)

print('model size {}'.format(get_model_parameters(model)))
model.cuda()

if args.optimizer.lower() == 'adam':
    optimizer = apex.optimizers.FusedLAMB
elif args.optimizer.lower() == 'lamb':
    optimizer = apex.optimizers.FusedLAMB
else:
    optimizer = apex.optimizers.FusedSGD

optim = optimizer(
    model.parameters(),
    eps=2.0e-7,
    lr=1.0e-7,
)
model, optim = amp.initialize(model, optim, opt_level="O2", verbosity=0)
model = apex.parallel.DistributedDataParallel(model)

warm_up_steps = 500
lr_opt_steps = max_learning_rate / 1000000
warm_up_lr_opt_steps = max_learning_rate / warm_up_steps


def metric_sum(val):
    tensor = torch.tensor(val).cuda()
    dist.reduce(tensor, 0)
    return tensor.item()


def metric_mean(val):
    tensor = torch.tensor(val).cuda()
    dist.reduce(tensor, 0)
    return tensor.item() / dist.get_world_size()


def get_shuffle_train_data():
    pool = {}
    for one in train_data:
        length = len(one[0]) // 5
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    whole_data = [x for y in length_lst for x in pool[y]]
    remove_data_size = len(whole_data) % dist.get_world_size()
    thread_data = [whole_data[x + args.local_rank] for x in
                   range(0, len(whole_data) - remove_data_size, dist.get_world_size())]
    return thread_data


def get_train_data(batch, max_len=doc_max_length_size):
    batch, _ = padding(batch, pads=tokenizer.pad_token_id, max_len=max_len)
    seq = batch.flatten()
    real_end_pos = np.where(seq == -1)[0]
    np.put(seq, real_end_pos, tokenizer.cls_token_id)
    all_end_pos_seq = np.where(seq == tokenizer.cls_token_id)[0]
    label = np.zeros(shape=len(all_end_pos_seq), dtype='float32')
    for i, j in enumerate(all_end_pos_seq):
        if j in real_end_pos:
            label[i] = 1
    batch = seq.reshape(batch.shape)
    return batch, label


current_number = 0
update_number = 0


def evaluation(epo):
    results = []
    for i in range(dist.get_world_size()):
        results.extend(load_file('{}.tmp.obj'.format(i)))
        os.remove('{}.tmp.obj'.format(i))
    print('epoch:{},total:{}'.format(epo, len(results)))
    threshold = 0.5
    precision, recall, f1, macro_f1, accuracy = evaluate_comqa(results, threshold)
    print('threshold:{}\nprecision:{}\nrecall:{}\nf1:{}\nmacro_f1:{}\naccuracy:{}\n{}'.format(
        threshold, precision,
        recall, f1,
        macro_f1, accuracy,
        '===' * 10))
    return [precision, recall, macro_f1, f1, accuracy]


def dev(epo):
    model.eval()
    total = len(thread_dev_data)
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size)):
            sample = thread_dev_data[i:i + batch_size]
            context_raw = [x[0] for x in sample]
            paras = [x[1] for x in sample]
            batch, label = get_train_data(context_raw, 512)
            batch = torch.LongTensor(batch)
            mask_idx = torch.eq(batch, tokenizer.cls_token_id)
            answer_logits = model([batch.cuda(), None])
            end_num = mask_idx.sum(1).data.numpy().tolist()
            answer_logits = answer_logits.cpu().data.numpy().tolist()
            start = 0
            for one_sent_end_num, para in zip(end_num, paras):
                pred = answer_logits[start:start + one_sent_end_num]
                results.append([pred, para])
                start += one_sent_end_num
    dump_file(results, '{}.tmp.obj'.format(dist.get_rank()))
    dist.barrier()
    if dist.get_rank() == 0:
        return evaluation(epo)
    return None


def train(epo):
    global current_number, update_number
    model.train()
    data = get_shuffle_train_data()
    total = len(data)
    total_loss = 0
    num = 0
    pre_time = None
    instance_number = 0
    for i in range(0, total, batch_size):
        context = [x[0] for x in data[i:i + batch_size]]
        batch, label = get_train_data(context)
        batch = torch.LongTensor(batch)
        loss = model([batch.cuda(), torch.FloatTensor(label).cuda()])
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        total_loss += loss.item() * len(context)
        instance_number += len(context)
        optim.step()
        optim.zero_grad()
        update_number += 1
        for param_group in optim.param_groups:
            if update_number > warm_up_steps:
                param_group['lr'] -= lr_opt_steps
            else:
                param_group['lr'] += warm_up_lr_opt_steps
        num += 1
        if num % args.log_interval == 0:
            if pre_time is None:
                eclipse = 0
            else:
                eclipse = time.time() - pre_time
            total_loss = metric_sum(total_loss)
            instance_number = metric_sum(instance_number)
            if dist.get_rank() == 0:
                print(
                    'epoch {}, mask loss is {:5.4f}, ms per batch is {:7.4f}, eclipse {:4.3f}%  lr={:e}'.format(epo,
                                                                                                                total_loss / instance_number,
                                                                                                                1000 * eclipse / instance_number,
                                                                                                                i * 100 / total,
                                                                                                                optim.param_groups[
                                                                                                                    0][
                                                                                                                    'lr']))
            pre_time = time.time()
            total_loss = 0
            instance_number = 0


if __name__ == '__main__':
    results = []
    best_f1 = 0
    for i in range(args.epoch):
        train(i)
        results = dev(i)
        output = {}
        if dist.get_rank() == 0:
            print('epoch {} done!! result is {}'.format(i, results))
            if results[2] > best_f1:
                best_f1 = results[2]
                for name, param in model.module.named_parameters():
                    output[name] = param.data.cpu().numpy()
                dump_file(output, args.model_save_path)
