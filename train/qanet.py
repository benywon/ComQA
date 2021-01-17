# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/17 下午9:14
 @FileName: qanet.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import sys
import time
import apex
import torch
import torch.distributed as dist
from apex import amp

sys.path.append('..')
from modules.QANet import QANet
from train.parser import get_argument_parser

from utils import *

np.random.seed(1000)
torch.manual_seed(1024)
torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
args = get_argument_parser()
print(args.local_rank, dist.get_rank(), dist.get_world_size())
torch.cuda.set_device(args.local_rank)

vocab_size = 50000
n_embedding = 128
n_hidden = 256
n_layer = 4
n_head = 4
batch_size = 32

max_learning_rate = 4e-4
doc_max_length_size = 1024

train_data = load_file(args.train_file_path)
dev_data = load_file(args.dev_file_path)
dev_data = sorted(dev_data, key=lambda x: len(x[0] + x[1]))
remove_data_size = len(dev_data) % dist.get_world_size()
thread_dev_data = [dev_data[x + args.local_rank] for x in
                   range(0, len(dev_data) - remove_data_size, dist.get_world_size())]

print('train data size is {} test size {}'.format(len(train_data), len(dev_data)))
model = QANet(vocab_size, n_embedding, n_hidden, n_layer, n_head)
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
        length = len(one[0] + one[1]) // 5
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
    batch, _ = padding(batch, max_len=max_len)
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
            question_raw = [x[0] for x in sample]
            context_raw = [x[1] for x in sample]
            paras = [x[2] for x in sample]
            batch, label = get_train_data(context_raw, 1024)
            batch = torch.LongTensor(batch)
            question, _ = padding(question_raw, max_len=32)
            question = torch.LongTensor(question).cuda()
            mask_idx = torch.eq(batch, vocab_size)
            answer_logits = model(question, batch.cuda(), None)
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
        question = [x[0] for x in data[i:i + batch_size]]
        context = [x[1] for x in data[i:i + batch_size]]
        question, _ = padding(question, max_len=32)
        context, label = get_train_data(context)
        label = torch.FloatTensor(label).cuda()
        question = torch.LongTensor(question).cuda()
        context = torch.LongTensor(context).cuda()
        loss = model(question, context, label)
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
