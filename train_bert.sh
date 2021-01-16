#!/bin/bash
base_dir=`pwd`
jobname='bert for ComQA'
train_features_path=${base_dir}/data/train.bert.obj
dev_features_path=${base_dir}/data/dev.bert.obj
test_features_path=${base_dir}/data/test.bert.obj
model_save_path=${base_dir}/model/bert.comqa.base.th
echo $jobname
echo "start processing data"
echo $train_features_path
echo $dev_features_path
echo $test_features_path

cd process
python3 process_bert.py --train=${train_features_path} --dev=${dev_features_path} --test=${test_features_path}
cd ../train

echo "start training"
python3.6 -m torch.distributed.launch --nproc_per_node=4 bert.py \
--train_file_path=${train_features_path} \
--dev_file_path=${dev_features_path} \
--model_save_path=${model_save_path} \
--epoch=10 \
--pretrain_model=${base_dir}/model/bert.base.th

cd ../evaluation
echo "start evaluation"
python3 evaluate_bert.py ${test_features_path} ${model_save_path}