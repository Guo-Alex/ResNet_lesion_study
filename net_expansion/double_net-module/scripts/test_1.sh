#!/bin/sh

#prototxt='prototxt/double_CNN_bn-conv32_3-conv32_to_16.prototxt'
#caffemodel='caffemodel/CNN_bn-conv32_3-conv32_to_16.caffemodel'
prototxt='prototxt/double_CNN_bn.prototxt'
caffemodel='caffemodel/double_CNN_bn.caffemodel'
gpu=0

/home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt --weights $caffemodel --gpu $gpu --iterations 10000 2>&1 |tee log/double_CNN_bn.log 
