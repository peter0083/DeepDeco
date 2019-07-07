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
from _datetime import datetime, timezone
import os
import imageio


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

parser.add_argument('--speed', type=str,
                    help='Input style transfer speed. slow: 1 hour (best quality). medium: 20 min. fast: 10 min',
                    metavar='IN_SPEED', required=False)

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

currentDT = datetime.now(timezone.utc)

if input_text.speed == 'slow':
    timer = '3600'  # 60 sec/min * 60 min = 3600 sec

elif input_text.speed == 'medium':
    timer = '1200'  # 60 sec/min * 20 min = 1200 sec

else:
    timer = '600'

bashCommand40 = "timeout " + timer + " " \
                "python src/ftdeepphoto/run_fpst.py --in-path " \
                + input_text.content + " " \
                "--style-path " \
                "data/" + key_for_max_similarity + " --checkpoint-path checkpoints/ --out-path " \
                "output/output_stylized_image" + currentDT.strftime('%Y_%m_%d_%H_%M_%S') + ".jpg --deeplab-path " \
                "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz --slow"
print(bashCommand40)

start = time.time()
os.system(bashCommand40)
end = time.time()
print("style transfer time: ", end - start, " seconds")

# make a gif
images = []

for file in os.listdir("."):
    if file.endswith(".png"):
        images.append(imageio.imread(file))
imageio.mimsave("output/output_stylized_image" + currentDT.strftime('%Y_%m_%d_%H_%M_%S') + ".gif", images)

