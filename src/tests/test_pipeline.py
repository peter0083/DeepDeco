#!/usr/bin/env python

import pytest
import subprocess


def test_input_text_description():
    """
    This function checks string input type for texture description
    The parser function should raise a TypeError
    """

    inference_command0 = ["python", "inference_aws.py", "--input",
                         123, "--content", "data/images/chair_sketch.jpg"]

    with pytest.raises(TypeError):
        subprocess.run(inference_command0)


def test_input_content_image_path():
    """
    This function checks string input type for texture description
    The parser function should raise a TypeError
    """

    inference_command1 = ["python", "inference_aws.py", "--input",
                         "patterned glass", "--content", 123]

    with pytest.raises(TypeError):
        subprocess.run(inference_command1)

def test_input_content_image_path():
    """
    This function checks string input type for inference speed
    The parser function should raise a TypeError
    """

    inference_command2 = ["python", "inference_aws.py", "--input",
                         "patterned glass", "--content", "data/images/chair_sketch.jpg", "--speed", 0]

    with pytest.raises(TypeError):
        subprocess.run(inference_command2)