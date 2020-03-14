#! /bin/bash
#python create_data.py kitti_data_prep --root_path=/mnt/data2/Kitti_for_TANet_2/object

CUDA_VISIBLE_DEVICES=5 python ./pytorch/train.py train --config_path=./configs/tanet/car/xyres_16.config --model_dir=/mnt/data2/TANet_2/second.pytorch/second/open_source_train_car_tanet_standard_really_psa_weight_2  --refine_weight=2 #--multi_gpu=True
CUDA_VISIBLE_DEVICES=5 python ./pytorch/train.py evaluate --config_path=./configs/tanet/car/xyres_16.config --model_dir=/mnt/data2/TANet_2/second.pytorch/second/open_source_train_car_tanet_standard_really_psa_weight_2 --measure_time=True --batch_size=1