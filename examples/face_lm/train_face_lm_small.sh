#!/usr/bin/env sh
cd ../../

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/face_lm/face_lm_solver_small.prototxt

