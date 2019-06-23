##############################################
# this script runs inference
# to generate an image based on
# the following inputs
# 1. a text description of the object (MS AttnGAN)
# 2. a outline image of an object (Nvidia GauGAN)

# to run this script on AWS Linux Set up the data folder to contain four things:
#
# 'train' folder. This contains 'filenames.pickle' for the training files 'test' folder. This should contain
#  'filenames.pickle' for the test files
#
# 'coco' folder. This should contain all the images from the coco dataset. (train, val and test)
#
# 'captions.pickle' (if you plan to use pre-trained models)
#  Update your configuration files in 'code/cfg/' to point to the parent
#
# directory of the 'coco' folder. E.g: if 'coco' folder is in 'data/coco/coco' path, put the 'data' path in your
# config files as 'data/coco'
#
# ensure that `sudo python -V` gives your python 3.6
#
# ensure that `sudo conda install --file requirements.txt` is successful then execute "sudo python inference_aws.py"
# #############################################

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

# need a separate command to download vgg19 weight from S3


# add download for attnGAN in WK3

# Part 2
# image similarity search

# Part 3
# Deep Lab

# print("DeepLab V2 segmentation: setup conda environment")
#
# bashCommand30 = 'conda env create -f configs/conda_env.yaml'
#
# os.system(bashCommand30)
#
# bashCommand31 = 'conda activate deeplab-pytorch'
#
# os.system(bashCommand31)
#
# print("DeepLab V2 segmentation: download caffemodel pre-trained on ImageNet and 91-class COCO (1GB+)")
#
# bashCommand32 = 'sudo bash /home/ubuntu/DeepDeco/src/deeplab/scripts/setup_caffemodels.sh'
#
# os.system(bashCommand32)
#
# print("DeepLab V2 segmentation: Convert the caffemodel to pytorch compatible.")
#
# bashCommand33 = 'sudo python /home/ubuntu/DeepDeco/src/deeplab/convert.py --dataset coco'
#
# os.system(bashCommand33)
#
# print("DeepLab V2 segmentation: now segmenting image....")
#
# bashCommand34 = 'sudo python /home/ubuntu/DeepDeco/src/deeplab/demo.py single' \
#                  ' -c /home/ubuntu/DeepDeco/src/deeplab/configs/coco.yaml' \
#                  ' -m /home/ubuntu/DeepDeco/src/deeplab/data/models/coco/deeplabv1_resnet101/caffemodel' \
#                  '/deeplabv1_resnet101-coco.pth' \
#                  ' -i /home/ubuntu/DeepDeco/jeans.jpg'
#
# os.system(bashCommand34)
#
# print("DeepLab V2 segmentation: completed")


# Part 4
# Fast deep photo style transfer

print("Fast deep photo style transfer inference")


bashCommand40 = "python run_fpst.py --in-path /home/ubuntu/DeepDeco/ikea_vimle_series_sketch.jpg  --style-path " \
                "/home/ubuntu/DeepDeco/jeans.jpeg --checkpoint-path checkpoints --out-path " \
                "output/output_stylized_image.jpg --deeplab-path " \
                "deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz --batch-size 1 --slow "


os.system(bashCommand40)

