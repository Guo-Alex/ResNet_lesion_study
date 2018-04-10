#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --snapshot=snapshot/CNN_bn_30_iter_250001.solverstate --gpu 1 2>&1 | tee  ../log/xu_model_train_next_20170902.log  

