#!/usr/bin/env sh

/mnt/lgq/toolbars/caffe/build/tools/caffe train --solver=solver.prototxt --gpu 0 2>&1 | tee  ../logs/train/2018.03/xu_model_bn_60.log

