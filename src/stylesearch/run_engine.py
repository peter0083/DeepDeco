#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################
# adapted code from @IvonaTau
# https://github.com/IvonaTau/style-search
#
# #############################################

from argparse import ArgumentParser
import gensim
import pickle
import keras.applications.resnet50
from finder import initiate_engine, load
from training import CountVectModel
from search_engine import SearchEngine


# arguments for this script

# %% Parser function
def build_parser():
    """Parser function"""
    parser = ArgumentParser()

    # Input
    # text description of the furniture
    parser.add_argument('--input-text', type=str,
                        help='Input text description of the furniture',
                        metavar='IN_TEXT', required=True)

    # Default output path to same name as input in parent directory
    # parser.add_argument('--out-path', type=str,
    #                     dest='out_path', help='Output styled image path',
    #                     metavar='OUT_PATH', required=True)

    # Intermediate file save directories
    # parser.add_argument('--resized-dir', type=str,
    #                     dest='resized_dir', help='Resized image directory',
    #                     metavar='RESIZED_DIR', default=resized_dir)

    # parser.add_argument('--seg-dir', type=str,
    #                     dest='seg_dir', help='Segmented image directory',
    #                     metavar='SEG_DIR', default=seg_dir)

    return parser


# load dict
with open('pickles/products_dict.p', 'rb') as handle:
    products_dict = pickle.load(handle)
 
# load w2vec model
with open('pickles/word2vec_model.p', 'rb') as f:
    model = pickle.load(f)


# build search engine
vectorizer = CountVectModel(products_dict)
search_engine = SearchEngine(products_dict, vectorizer, model)

model_extension = 'resnet'

# load cnn features
try:
    with open('pickles/gallery_cnn_features_' + model_extension + '.pickle', 'rb') as f:
        GALLERY_FEATURES = pickle.load(f)
        print('Gallery CNN features loaded!')
except FileNotFoundError:
    print('No CNN features for model', model_extension)


print('Model extension is', model_extension)

# create model
model = keras.applications.resnet50.ResNet50(include_top=False)

print('Keras ResNet50 model created')
 

try:
    with open('pickles/clock_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_CLOCK = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_CLOCK = initiate_engine(app.config['CLOCK_DIR'], model_extension, load(app.config['VOC_CLOCK']))
try:
    with open('pickles/bed_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_BED = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_BED = initiate_engine(app.config['BED_DIR'], model_extension, load(app.config['VOC_BED']))
try:
    with open('pickles/chair_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_CHAIR = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_CHAIR = initiate_engine(app.config['CHAIR_DIR'], model_extension, load(app.config['VOC_CHAIR']))
try:
    with open('pickles/plant_pot_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_POT = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_POT = initiate_engine(app.config['POT_DIR'], model_extension, load(app.config['VOC_CLOCK']))
try:
    with open('pickles/sofa_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_SOFA = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    print ('No pickle file found')
    ENGINE_SOFA = initiate_engine(app.config['SOFA_DIR'], model_extension, load(app.config['VOC_SOFA']))
try:
    with open('pickles/table_' + model_extension + '.pickle', 'rb') as handle:
        ENGINE_TABLE = pickle.load(handle)
        print(handle, 'loaded')
except FileNotFoundError:
    ENGINE_TABLE = initiate_engine(app.config['TABLE_DIR'], model_extension, load(app.config['VOC_TABLE']))

print('All engines initiated')