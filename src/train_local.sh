#!/bin/bash

# python train.py --name [experiment_name] --dataset_mode custom --label_dir [path_to_labels] -- image_dir [path_to_images] --label_nc [num_labels]

# running locally on Peter's own machine
# ie.1 use testrun for experiment_name
# ie.2 use /Users/peterlin/DeepDeco/dataset/datasets_mini copy/coco_stuff/val_label for label_dir
# ie.3 use /Users/peterlin/DeepDeco/dataset/datasets_mini copy/coco_stuff/val_img for image_dir
# ie.4 use 200 for label_nc which means number of label classes
# training requires GPU

python /home/ubuntu/DeepDeco/train_gaugan.py --name "$1" --dataset_mode custom --label_dir "$2" --image_dir "$3" --label_nc $4