#!/usr/bin/env sh


TOOLS=../../build/tools

#$TOOLS/caffe train  --solver=solver32_two_stages_lowres.prototxt2 &> log_two_stages2.txt

$TOOLS/caffe train  --solver=portrait_solver_one_stage5.prototxt

#$TOOLS/caffe train  --solver=solver32_two_stages_lowres.prototxt4 &> log_two_stages4.txt

#$TOOLS/caffe train  --solver=solver32_two_stages.prototxt2 &> log_two_stages2.txt

#$TOOLS/caffe train  --solver=solver32_two_stages.prototxt3 &> log_two_stages3.txt



