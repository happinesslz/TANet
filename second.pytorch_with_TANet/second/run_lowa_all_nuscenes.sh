#! /bin/bash
python create_data.py nuscenes_data_prep --root_path=/mnt/data4/NuScenes/train --version="v1.0-trainval" --max_sweeps=10 --dataset_name="NuScenesDataset"
python create_data.py nuscenes_data_prep --root_path=/mnt/data4/NuScenes/test --version="v1.0-test" --max_sweeps=10 --dataset_name="NuScenesDataset"

CUDA_VISIBLE_DEVICES=5 python ./pytorch/train.py train --config_path=./configs/nuscenes/all.pp.lowa.config --model_dir=/mnt/data2/TANet_2/second.pytorch/second/train_all_lowa_nuscenes # --multi_gpu=True
CUDA_VISIBLE_DEVICES=5 python ./pytorch/train.py evaluate  --config_path=./configs/nuscenes/all.pp.lowa.config --model_dir=/mnt/data2/TANet_2/second.pytorch/second/train_all_lowa_nuscenes --measure_time=True --batch_size=1