#!/bin/bash

# Download released caffemodels
wget -nc -P /home/ubuntu/DeepDeco/src/deeplab/data http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip

echo ===============================================================================================
echo "Next, try unzipping prototxt_and_model.zip:"
echo ===============================================================================================

unzip -n /home/ubuntu/DeepDeco/src/deeplab/data/prototxt_and_model.zip -d /home/ubuntu/DeepDeco/src/deeplab/data/

# Move caffemodels to data directories
## MSCOCO
mv /home/ubuntu/DeepDeco/src/deeplab/data/init.caffemodel /home/ubuntu/DeepDeco/src/deeplab/data/models/coco/deeplabv1_resnet101/caffemodel
## PASCAL VOC 2012
mv /home/ubuntu/DeepDeco/src/deeplab/data/train_iter_20000.caffemodel /home/ubuntu/DeepDeco/src/deeplab/data/models/voc12/deeplabv2_resnet101_msc/caffemodel
mv /home/ubuntu/DeepDeco/src/deeplab/data/train2_iter_20000.caffemodel /home/ubuntu/DeepDeco/src/deeplab/data/models/voc12/deeplabv2_resnet101_msc/caffemodel

echo ===============================================================================================
echo "Next, try running script below:"
echo "convert.py"
echo ===============================================================================================