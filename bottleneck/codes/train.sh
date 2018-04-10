#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --gpu 2 2>&1 | tee  ../log/xu_model_bottleneck_v2_20170921.log

