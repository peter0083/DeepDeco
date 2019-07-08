#!/usr/bin/env python

import pytest
import subprocess


def test_input_content_image_path():
    """
    This function checks string input type for content image path
    The parser function should raise a TypeError
    """

    deep_photo_command0 = ["python", "src/ftdeepphoto/run_fpst.py", "--in-path",
                         123, "--style-path", "data/images/000.171.33.jpg", "--checkpoint-path", "checkpoints", "--out-path",
                         "output/output_stylized_image.jpg", "--deeplab-path",
                         "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz",
                         "--slow"]

    with pytest.raises(TypeError):
        subprocess.run(deep_photo_command0)


def test_input_style_path():
    """
    This function checks string input type for style image path
    The parser function should raise a TypeError
    """

    deep_photo_command1 = ["python", "src/ftdeepphoto/run_fpst.py", "--in-path",
                         "data/images/chair_sketch.jpeg", "--style-path", 123, "--checkpoint-path", "checkpoints", "--out-path",
                         "output/output_stylized_image.jpg", "--deeplab-path",
                         "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz",
                         "--slow"]

    with pytest.raises(TypeError):
        subprocess.run(deep_photo_command1)


def test_deeplab_path():
    """
    This function checks string input type for deeplab pretrained model path
    The parser function should raise a TypeError
    """

    deep_photo_command2 = ["python", "src/ftdeepphoto/run_fpst.py", "--in-path",
                         "data/images/chair_sketch.jpeg", "--style-path", "data/images/000.171.33.jpg", "--checkpoint-path", "checkpoints", "--out-path",
                         "output/output_stylized_image.jpg", "--deeplab-path",
                         123,
                         "--slow"]

    with pytest.raises(TypeError):
        subprocess.run(deep_photo_command2)