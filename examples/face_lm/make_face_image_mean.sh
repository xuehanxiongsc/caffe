#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/face_lm
TOOLS=build/tools
DATA=examples/face_lm

$TOOLS/compute_image_mean --logtostderr=1 $DATA/face_image_train_small_lmdb \
  $EXAMPLE/image_mean.binaryproto

$TOOLS/compute_image_mean --logtostderr=1 $DATA/face_lm_train_small_lmdb \
  $EXAMPLE/landmark_mean.binaryproto
echo "Done."
