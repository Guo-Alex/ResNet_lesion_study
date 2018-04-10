#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --gpu 4 2>&1 | tee  ../log/xu_model_bn_30_train_20170905.log 

