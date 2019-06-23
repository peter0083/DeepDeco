#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:50:02 2018

@author: hyang
"""
# coding: utf-8

# Segment input and style photos using DeepLab
# Modified from DeepLab demo Jupyter notebook
# %% Import

import os
#from io import BytesIO
import tarfile
#from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

# %% Helper methods

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    # Run on CPU to conserve GPU memory for later operations in style transfer pipeline
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    self.sess = tf.Session(graph=self.graph, config=config)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

# %% Load pretrained model

#model_path = os.path.join(opts.model_dir, 'deeplab_model.tar.gz')
def loadModel(model_path):
    MODEL = DeepLabModel(model_path)
    print('DeepLab model %s loaded successfully!' % model_path)
    return MODEL

# %% Visualization functions

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FROM DEEPLAB DEMO -- RUN ON URL !!!!!!
#@title Run on sample images {display-mode: "form"}

#SAMPLE_IMAGE = 'image2'  # @param ['image1', 'image2', 'image3']
#IMAGE_URL = ''  #@param {type:"string"}
#
#_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
#              'deeplab/g3doc/img/%s.jpg?raw=true')

#def run_visualization(url):
#  """Inferences DeepLab model and visualizes result."""
#  try:
#    f = urllib.request.urlopen(url)
#    jpeg_str = f.read()
#    original_im = Image.open(BytesIO(jpeg_str))
#  except IOError:
#    print('Cannot retrieve image. Please check url: ' + url)
#    return
#
#  print('running deeplab on image %s...' % url)
#  resized_im, seg_map = MODEL.run(original_im)
#
#  vis_segmentation(resized_im, seg_map)
#  return(resized_im, seg_map)

def run_visualization_local(imagePath, imageName, resized_dir, seg_dir, MODEL):
  """Inferences DeepLab model and visualizes result."""
  try:
#    jpeg_str = f.read()
    original_im = Image.open(imagePath)
  except IOError:
    print('Cannot retrieve image. Please check path: ' + imagePath)
    return

  print('running deeplab on image %s...' % imageName)
  resized_im, seg_map = MODEL.run(original_im)

  # Save
  type(resized_im)
  resized_im.save(os.path.join(resized_dir, imageName))
  type(seg_map)
  seg_map[seg_map > 1] = 255
  segIm = Image.fromarray(seg_map.astype('uint8')) # Temporary binary classification
  segIm.save(os.path.join(seg_dir, imageName))
  
  # Visualize
  # TO DO: Trigger this from a flag
#  vis_segmentation(resized_im, seg_map)
# %% Main
def main(model_path, target_dir, target_fname, resized_dir, seg_dir):
    
    # Using mobilenetv2, trained on COCO and VOC 2012
    # TODO: retrain on ADE20K, interior objects subset
    #MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
    MODEL = loadModel(model_path)
    
    # Run visualization and save preprocessed input and style images (resized, segmentation maps)
    run_visualization_local(target_dir, target_fname, resized_dir, seg_dir, MODEL)
#    run_visualization_local(opts, opts.in_path, opts.inputFileName, MODEL)
#    run_visualization_local(opts, opts.style_path, opts.styleFileName, MODEL)

if __name__ == '__main__':
    main()
