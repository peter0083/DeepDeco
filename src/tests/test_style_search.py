#!/usr/bin/env python

import pytest
import subprocess


def test_input_text():
    """
    This function checks string input type
    The Find-Max-Similarity function should raise a TypeError
    """

    run_engine_command = ["python", "src/stylesearch/run_engine.py", "--input",
                         123, "--content", "data/images/chair_sketch.jpeg"]

    with pytest.raises(TypeError):
        subprocess.run(run_engine_command)


def test_input_image():
    """
    This function checks string input type for image file path
    The Find-Max-Similarity function should raise a TypeError
    """

    run_engine_command = ["python", "src/stylesearch/run_engine.py", "--input",
                          "patterned glass", "--content", 123]

    with pytest.raises(TypeError):
        subprocess.run(run_engine_command)


