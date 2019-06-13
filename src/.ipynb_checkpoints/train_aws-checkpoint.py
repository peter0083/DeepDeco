import os
import sys
import subprocess

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]

subprocess.call(['python /home/ubuntu/DeepDeco/train_gaugan.py --name ', 
                 arg1, 
                 ' --dataset_mode custom --label_dir ', 
                 arg2, 
                 ' --image_dir ', 
                 arg3, 
                 ' --label_nc ', 
                 arg4])