# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午5:35
 @FileName: parser.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse


def get_argument_parser(return_args=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_file_path',
        type=str,
        required=True,
        help=
        'the training file path'
    )
    parser.add_argument(
        '--dev_file_path',
        type=str,
        required=True,
        help=
        'the dev file path'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        required=True,
        help=
        'which directory to save the model'
    )
    parser.add_argument(
        '--pretrain_model',
        type=str,
        help=
        'the path to the pretrained model if it requires pretraining'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=20,
        help=
        'the training file path'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=500,
        help=
        'the logging interval for each'
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument(
        "--optimizer",
        type=str,
        default='Adam',
        help=
        "Whether to use cpu optimizer for training"
    )
    if return_args:
        args = parser.parse_args()
        return args
    return parser
