#!/bin/bash
base_dir=`pwd`
jobname='official bert for ComQA'
train_features_path=${base_dir}/data/train.official.bert.obj
dev_features_path=${base_dir}/data/dev.official.bert.obj
test_features_path=${base_dir}/data/test.official.bert.obj
model_save_path=${base_dir}/model/bert.comqa.official.th
echo $jobname
echo $train_features_path
echo $dev_features_path
echo $test_features_path
echo "start processing data"
cd process
python3 process_official_bert.py --train=${train_features_path} --dev=${dev_features_path} --test=${test_features_path}

cd ../train
echo "start training"
python3.6 -m torch.distributed.launch --nproc_per_node=4 official_bert.py \
--train_file_path=${train_features_path} \
--dev_file_path=${dev_features_path} \
--epoch=10 \
--model_save_path=${model_save_path}

cd ../evaluation
echo "start evaluation"
python3 evaluate_official_bert.py ${test_features_path} ${model_save_path}

