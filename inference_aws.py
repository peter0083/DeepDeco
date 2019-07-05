##############################################
# this script runs inference
# to generate an image based on
# the following inputs:
# 1. an image of a piece of furniture (sketch)
# 2. a string describing the desired text
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




# Part 2
# Fast deep photo style transfer

print("Fast deep photo style transfer inference")

bashCommand40 = "python src/ftdeepphoto/run_fpst.py --in-path " \
                "office_chair_sketch.jpeg " \
                "--style-path " \
                "ikea_timsfors.jpg --checkpoint-path checkpoints/ --out-path " \
                "output/output_stylized_image2.jpg --deeplab-path " \
                "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz --slow"

os.system(bashCommand40)

