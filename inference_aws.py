#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# this script runs inference
# to generate an image based on
# the following inputs:
# 1. an image of a piece of furniture (sketch)
# 2. a string describing the desired text
"""


import time
from argparse import ArgumentParser
from src.stylesearch.run_engine import StyleSearch
import pickle
import datetime
import os


start = time.time()


# arguments for this script
parser = ArgumentParser()

# Input
# text description of the furniture
parser.add_argument('--input', type=str,
                    help='Input text description of the furniture',
                    metavar='IN_TEXT', required=True)

parser.add_argument('--content', type=str,
                    help='Input sketch image file including its file path',
                    metavar='IN_CONTENT', required=True)

input_text = parser.parse_args()


# Part 2
# Style Search Engine

# load dict
with open('src/stylesearch/pickles/img2vec_dict.p', 'rb') as handle:
    img2vec_dict = pickle.load(handle)

ss = StyleSearch()
max_similarity, key_for_max_similarity = ss.find_max_similarity(input_text.input)

# Part 3
# Fast deep photo style transfer

print("Fast deep photo style transfer inference")

currentDT = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

bashCommand40 = "timeout 500 "\
                "python src/ftdeepphoto/run_fpst.py --in-path " \
                + input_text.content + " " \
                "--style-path " \
                "data/"+ key_for_max_similarity + " --checkpoint-path checkpoints/ --out-path " \
                "output/ --deeplab-path " \
                "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
print(bashCommand40)

start = time.time()
os.system(bashCommand40)
end = time.time()
print("style transfer time: ", end - start, " seconds")

