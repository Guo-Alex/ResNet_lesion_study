#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --snapshot=snapshot/CNN_bn_bottleneck_iter_250001.solverstate --gpu 1 2>&1 | tee  ../log/xu_model_bn_bottleneck_train_next_20170820.log

