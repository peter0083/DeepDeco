#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:22:34 2018

@author: hyang
"""
import os, errno
from argparse import ArgumentParser
import src.segmentDeepLab as seg
import fst
from src.utils import exists, list_files
from subprocess import call

# Main script to run fast deep photo style transfer

#%% Define defaults
main_dir = os.path.dirname(__file__)
deeplab_path = os.path.join(main_dir, 'deeplab/models/deeplab_model.tar.gz')
# Default folders for DeepLab
input_dir = os.path.join(main_dir, 'inputPhotos/')
resized_dir = os.path.join(main_dir, 'resizedPhotos/')
style_dir = os.path.join(main_dir + 'stylePhotos/')
seg_dir = os.path.join(main_dir + 'segMaps/')
output_dir = os.path.join(main_dir + 'outputPhotos/')
# Default folders for fast style transfer
VGG_PATH = 'fast-style-transfer-tf/data/imagenet-vgg-verydeep-19.mat' # point to deep photo weights
# FST options
BATCH_SIZE = 4
DEVICE = '/gpu:0'

# %% Parser function
def build_parser():
    """Parser function"""
    parser = ArgumentParser()

# Input and output
    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='Input image path',
                        metavar='IN_PATH', required=True)

    parser.add_argument('--style-path', type=str,
                        dest='style_path', help='Style image path',
                        metavar='STYLE_PATH', required=True)

    # Default output path to same name as input in parent directory
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help='Output styled image path',
                        metavar='OUT_PATH', required=True)
    
# Intermediate file save directories
    parser.add_argument('--resized-dir', type=str,
                        dest='resized_dir', help='Resized image directory',
                        metavar='RESIZED_DIR', default=resized_dir)

    parser.add_argument('--seg-dir', type=str,
                        dest='seg_dir', help='Segmented image directory',
                        metavar='SEG_DIR', default=seg_dir)

# Deep Lab
    parser.add_argument('--deeplab-path', type=str,
                        dest='deeplab_path', help='Path to DeepLab model',
                        metavar='DEEPLAB_path', default=deeplab_path)

# Fast style transfer
    parser.add_argument('--checkpoint-path', type=str,
                        dest='checkpoint_dir',
                        help='Directory containing checkpoint files',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')
    
    # Deep photo style transfer (slow)
    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='Original Luan approach (very slow)',
                        default=False)

    return parser
# %% Helper methods

def ensure_folders(directory):
    """ If directory doesn't exist, make directory """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def check_opts(opts):
    opts.resized_dir = os.path.abspath(opts.resized_dir)
    opts.seg_dir = os.path.abspath(opts.seg_dir)
    opts.deeplab_path = os.path.abspath(opts.deeplab_path)
    
    opts.inputFileName = opts.in_path.split('/')[-1]
    opts.styleFileName = opts.style_path.split('/')[-1]
    opts.checkpointName = opts.checkpoint_dir.split('/')[-1].split('.')[0]
    opts.resized_path = os.path.join(opts.resized_dir, opts.inputFileName)
    opts.resized_style_path = os.path.join(opts.resized_dir, opts.styleFileName)
    opts.seg_path = os.path.join(opts.seg_dir, opts.inputFileName)
    opts.seg_style_path = os.path.join(opts.seg_dir, opts.styleFileName)
    
    ensure_folders(input_dir)
    ensure_folders(resized_dir)
    ensure_folders(style_dir)
    ensure_folders(seg_dir)
    ensure_folders(output_dir)
    # !!! IF NAMES MATCH, THROW EXCEPTION TO PREVENT OVERWRITING !!!
    if opts.inputFileName == opts.styleFileName:
        raise ValueError('Input and style file names cannot be the same')
        
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    
    return opts

# Function to retrieve files from directory
def _get_files(img_dir):
    """List all files in directory"""
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

# %% Full pipeline for slow deep photo style transfer or fast photo style transfer
def main():
    parser = build_parser()
    opts = parser.parse_args()
    opts = check_opts(opts)
    
    if opts.slow:
        # Call DeepLab auto-segmentation
        # From tensorflow github with minor modifications:
        # https://github.com/tensorflow/models/tree/master/research/deeplab
        seg.main(opts.deeplab_path, opts.in_path, opts.inputFileName, opts.resized_dir, opts.seg_dir)
        seg.main(opts.deeplab_path, opts.style_path, opts.styleFileName, opts.resized_dir, opts.seg_dir)
        print("CALLING SLOW DEEP PHOTO STYLE")
        print("Slow: %s" % opts.slow)
        
        # Now call slow deep photo style transfer
        # From Louie Yang's github with minor modifications:
        # https://github.com/LouieYang/deep-photo-styletransfer-tf
        cmd = ['python', '-m', 'cProfile', '-o', 'deepPhotoProfile_Adams' \
        , 'deep-photo-styletransfer-tf/deep_photostyle.py', '--content_image_path' \
        , opts.resized_path, '--style_image_path', opts.resized_style_path \
        , '--content_seg_path', opts.seg_path, '--style_seg_path', opts.seg_style_path \
        , '--style_option', '2', '--output_image', opts.out_path \
        , '--max_iter', '10000', '--save_iter', '100', '--lbfgs']
        print(cmd)
        call(cmd)
    else:
        print("CALLING FAST STYLE TRANSFER")
        # Use lengstrom's fast style transfer network to perform style transfer using
        # checkpoint trained for photorealistic style transfer
        # From Logan Engstrom's github with major modifications:
        # https://github.com/lengstrom/fast-style-transfer
        fst.main(opts)

    call(['open' , opts.out_path])
    
if __name__ == '__main__':
    main()
