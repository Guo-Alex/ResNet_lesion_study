#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --snapshot=snapshot/CNN_bn_50_iter_250001.solverstate --gpu 2 2>&1 | tee  ../log/xu_model_bn_50_train_next_20170808.log  

