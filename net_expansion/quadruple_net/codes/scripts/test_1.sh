#!/bin/sh

prototxt='CNN_bn_50.prototxt'
caffemodel='snapshot/CNN_bn_50_iter_10000.caffemodel'
gpu=0

/home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt --weights $caffemodel --gpu $gpu --iterations 500000 2>&1 |tee ../log/test/CNN_bn_50_batch1_iter_10000.log
