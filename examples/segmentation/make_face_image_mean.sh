#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/segmentation
TOOLS=build/tools
DATA=examples/segmentation

$TOOLS/compute_image_mean --logtostderr=1 $DATA/segmentation_image_train_lmdb
echo "Done."
