#!/bin/bash

# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 24 --modes 5 &
# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 24 --modes 10 &
# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 24 --modes 20 &
# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 24 --modes 30 &
# wait

# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 48 --modes 5 &
# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 48 --modes 10 &
# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 48 --modes 20 &
# python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 48 --modes 30 &
# wait

python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 64 --modes 5 
python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 64 --modes 10 
python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 64 --modes 20 
python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 64 --modes 30 

python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 80 --modes 5 
python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 80 --modes 10 
python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 80 --modes 20 
python train_mesoscale_width_modes.py --config_path configs/pretrain/mesoscale/width_modes.yaml --width 80 --modes 30 

wait