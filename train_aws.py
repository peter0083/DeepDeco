##############################################
# this script downloads the data from
# then runs the train scripts to train
# the following models in the order listed
# 1. Nvidia GauGan
# 2. MS AttnGAN

# to run this script on AWS Linux
# ensure that `sudo python -V` gives your python 3.6
# ensure that `sudo pip install -r requirements.txt` is successful
# then
# execute "sudo python train_aws.py"
##############################################

import os
import boto3

# Part 1
# download data from S3


# code adapted from https://stackoverflow.com/questions/49772151/boto3-download-folder-from-s3
# and https://www.mydatahack.com/comprehensive-guide-to-download-files-from-s3-with-python/


def download_directory_from_s3(bucket_name,
                               remote_directory_name):

    """
    :param bucket_name: string
    :param remote_directory_name: string
    :return: local_path
    """

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for key in bucket.objects.filter(Prefix=remote_directory_name):
        if not os.path.exists(os.path.dirname(key.key)):
            os.makedirs(os.path.dirname(key.key))
        bucket.download_file(key.key, key.key)
        print('Downloaded file with boto3 resource')
    local_path = os.path.dirname(key.key)
    return local_path


download_directory_from_s3('gauganspade', 'datasets_mini')

# Part 2-1
# train Nvidia GauGAN

# to be added: a way to programmatically find the directory then save files

# for local training (need GPU)
# train gaugan path: /Users/peterlin/DeepDeco/src/gaugan/train_gaugan.py
# label path: /Users/peterlin/DeepDeco/datasets_mini copy/coco_stuff/val_label
# image path: /Users/peterlin/DeepDeco/datasets_mini copy/coco_stuff/val_img
# --no_instance
# number of classes: class labels cannot in the range [0, n_class - 1 ] leave as default 182

# full command python /Users/peterlin/DeepDeco/src/gaugan/train_gaugan.py --name "local-testrun" --dataset_mode ' \
#               'custom --label_dir "/Users/peterlin/DeepDeco/datasets_mini copy/coco_stuff/val_label" --image_dir ' \
#               '"/Users/peterlin/DeepDeco/datasets_mini copy/coco_stuff/val_img" --gpu_ids 0'


# for AWS training
# train gaugan path: /home/ubuntu/DeepDeco/src/gaugan/train_gaugan.py
# label path: /home/ubuntu/DeepDeco/datasets_mini/coco_stuff/val_label
# image path: /home/ubuntu/DeepDeco/datasets_mini/coco_stuff/val_img
# --no_instance
# number of classes: class labels cannot in the range [0, n_class - 1 ] leave as default 182
# full command python /home/ubuntu/DeepDeco/src/gaugan/train_gaugan.py --name "local-testrun" --dataset_mode ' \
#               'custom --label_dir "/home/ubuntu/DeepDeco/datasets_mini/coco_stuff/val_label" --image_dir ' \
#               '"/home/ubuntu/DeepDeco/datasets_mini/coco_stuff/val_img" --label_nc --no_instance --gpu_ids 0'

print("training Nvidia GauGAN")

bashCommand = 'sudo python /home/ubuntu/DeepDeco/src/gaugan/train_gaugan.py --name "local-testrun" --dataset_mode ' \
               'custom --label_dir "/home/ubuntu/DeepDeco/datasets_mini/coco_stuff/val_label" --image_dir ' \
               '"/home/ubuntu/DeepDeco/datasets_mini/coco_stuff/val_img" --no_instance --gpu_ids 0'
print("running bash command")

os.system(bashCommand)