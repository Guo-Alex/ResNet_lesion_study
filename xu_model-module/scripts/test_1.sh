#!/bin/sh

prototxt='prototxt/CNN_bn-conv32_3-conv32_to_16.prototxt'
caffemodel='caffemodel/CNN_bn-conv32_3-conv32_to_16.caffemodel'
gpu=1

/home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt --weights $caffemodel --gpu $gpu --iterations 12500 2>&1 |tee log/CNN_bn-conv32_3-conv32_to_16.log 
