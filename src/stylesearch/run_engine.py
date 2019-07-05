#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################
# img_to_text pickle and inspiration were
# from @IvonaTau
# adapted code from @IvonaTau
# https://github.com/IvonaTau/style-search
#
# #############################################

from argparse import ArgumentParser
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
from train import Vectorizer
import time
import boto3
import os


# arguments for this script
parser = ArgumentParser()

# Input
# text description of the furniture
parser.add_argument('--input', type=str,
                    help='Input text description of the furniture',
                    metavar='IN_TEXT', required=True)

parser.add_argument('--weight_path', type=str,
                    help='path to the pre-trained Word2Vec weight file',
                    metavar='IN_TEXT', required=False)

input_text = parser.parse_args()

# load dict
with open('pickles/img2vec_dict.p', 'rb') as handle:
    img2vec_dict = pickle.load(handle)


class StyleSearch:

    # Optional
    # download data from S3
    def download_directory_from_s3(self, bucket_name,
                                   remote_directory_name):
        """
        To download sample data from S3 bucket,
        AWS CLI credential is required.

        :param bucket_name: string
        :param remote_directory_name: string
        :return: local_path
        """
        # code adapted from https://stackoverflow.com/questions/49772151/boto3-download-folder-from-s3
        # and https://www.mydatahack.com/comprehensive-guide-to-download-files-from-s3-with-python/

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        for key in bucket.objects.filter(Prefix=remote_directory_name):
            if not os.path.exists(os.path.dirname(key.key)):
                os.makedirs(os.path.dirname(key.key))
            bucket.download_file(key.key, key.key)
            print('Downloaded file with boto3 resource')
        local_path = os.path.dirname(key.key)
        return local_path

    # core function
    def find_max_similarity(self, text):
        """
        This function returns the style image that best matches the text input.
        :param text: string, description of desired style/texture
        :return: max_similarity [float], cosine similarity score of the closest style image
                 key_for_max_similarity [string], file path and file name of the closest style image
        """
        start = time.time()

        regex = " *[%s]+ *" % string.punctuation.replace("\\", "\\\\").replace("]", "\\]")
        stop_list = set('for a of the and to in view more product information \
                        an w very by has ikea get with as information you it on that have \
                        price reflects selected options guarantee brochure year read about terms'.split())
        text = text.lower().split(" ")
        text = [word for word in text if word not in stop_list]
        text = ' '.join(text)
        text = re.sub(regex, " ", text)
        input_vec = Vectorizer.sentence2vec(text.split(" "))
        max_similarity = 0
        key_for_max_similarity = None
        for key, value in img2vec_dict.items():
            if max_similarity >= cosine_similarity(img2vec_dict[key], input_vec):
                continue
            else:
                max_similarity = cosine_similarity(img2vec_dict[key], input_vec)
                key_for_max_similarity = key

        # retrieve image
        print("style description: ", input_text.input)

        print("cosine similarity score: ", max_similarity)
        print("style image found: ", key_for_max_similarity)
        end = time.time()
        print("style search time: ", end - start, " seconds")

        return max_similarity, key_for_max_similarity

ss = StyleSearch()
ss.find_max_similarity(text=input_text)

