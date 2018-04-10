#!/usr/bin/env sh

/home/lgq/Workspace/caffe/build/tools/caffe train --solver=solver.prototxt --gpu 2 2>&1 | tee  ../log/xu_model_bn_40_train_20170907.log

