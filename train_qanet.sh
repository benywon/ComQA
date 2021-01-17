#!/bin/bash
base_dir=`pwd`
jobname='QANet for ComQA'
train_features_path=${base_dir}/data/train.qanet.obj
dev_features_path=${base_dir}/data/dev.qanet.obj
test_features_path=${base_dir}/data/test.qanet.obj
model_save_path=${base_dir}/model/qanet.comqa.th

echo $jobname
echo "start processing data"
echo $train_features_path
echo $dev_features_path
echo $test_features_path

cd process
python3 process_qanet.py --train=${train_features_path} --dev=${dev_features_path} --test=${test_features_path}
cd ../train

echo "start training"
export OMP_NUM_THREADS=2
python3.6 -m torch.distributed.launch --nproc_per_node=4 qanet.py \
--train_file_path=${train_features_path} \
--dev_file_path=${dev_features_path} \
--model_save_path=${model_save_path} \
--epoch=10

cd ../evaluation
echo "start evaluation"
python3 evaluate_qanet.py ${test_features_path} ${model_save_path}