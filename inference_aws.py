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


start = time.time()


# arguments for this script
parser = ArgumentParser()

# Input
# text description of the furniture
parser.add_argument('--input_style_text', type=str,
                    help='Input text description of the furniture',
                    metavar='IN_TEXT', required=True)


input_text = parser.parse_args()


# Part 2
# Style Search Engine

# load dict
with open('src/stylesearch/pickles/img2vec_dict.p', 'rb') as handle:
    img2vec_dict = pickle.load(handle)

ss = StyleSearch()
max_similarity, key_for_max_similarity = ss.find_max_similarity(input_text.input_style_text)

# Part 3
# Fast deep photo style transfer

print("Fast deep photo style transfer inference")

bashCommand40 = "python src/ftdeepphoto/run_fpst.py --in-path " \
                "office_chair_sketch.jpeg " \
                "--style-path " \
                "data/"+ key_for_max_similarity + " --checkpoint-path checkpoints/ --out-path " \
                "output/output_stylized_image2.jpg --deeplab-path " \
                "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz --slow"
print(bashCommand40)
# os.system(bashCommand40)

