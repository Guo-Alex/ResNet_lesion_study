#!/usr/bin/env sh

/mnt/lgq/toolbars/caffe/build/tools/caffe train --solver=solver.prototxt --snapshot=snapshot/CNN_bn_60_iter_50000.solverstate --gpu 0 2>&1 | tee  ../logs/train/2018.03/xu_model_bn_60_from_iter50000.log

