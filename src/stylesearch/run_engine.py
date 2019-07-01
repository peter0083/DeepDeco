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
from train import sentence2vec


def build_parser():
    """Parser function"""

    # arguments for this script
    parser = ArgumentParser()

    # Input
    # text description of the furniture
    parser.add_argument('--input', type=str,
                        help='Input text description of the furniture',
                        metavar='IN_TEXT', required=True)
    return parser


input_text = parser.parse_args()

# load dict
with open('pickles/img2vec_dict.p', 'rb') as handle:
    img2vec_dict = pickle.load(handle)


# load w2vec model
def find_max_similarity(text):
    regex = " *[%s]+ *" % string.punctuation.replace("\\", "\\\\").replace("]", "\\]")
    stop_list = set('for a of the and to in view more product infromation \
                    an w very by has ikea get with as information you it on thats have \
                    price reflects selected options guarantee  brochure year read about terms'.split())
    text = text.lower().split(" ")
    text = [word for word in text if word not in stop_list]
    text = ' '.join(text)
    text = re.sub(regex, " ", text)
    input_vec = sentence2vec(text.split(" "))
    max_similarity = 0
    key_for_max_similarity = None
    for key, value in img2vec_dict.items():
        if max_similarity >= cosine_similarity(img2vec_dict[key], input_vec):
            continue
        else:
            max_similarity = cosine_similarity(img2vec_dict[key], input_vec)
            key_for_max_similarity = key
    return max_similarity, key_for_max_similarity


# retrieve image
find_max_similarity(input_text.input)

print(max_similarity, key_for_max_similarity)
