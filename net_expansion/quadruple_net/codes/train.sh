#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --gpu 2 2>&1 | tee  ../log/train/xu_model_bn_50_train_`date +%Y%m%d`.log  

