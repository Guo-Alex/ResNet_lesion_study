#!/bin/sh

term='downsampling/'
#term='skip/'
prototxt="prototxt/$term"
caffemodel="caffemodel/$term"
all_prototxt=`ls $prototxt`
gpu=1

for file in $all_prototxt
do
    #    let 'gpu=count/4'
    prototxt_file=$prototxt$file
    delete_suffix=`echo $file |cut -d '.' -f 1`
    caffemodel_file=$caffemodel$delete_suffix'.caffemodel'
    /home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt_file --weights $caffemodel_file --gpu $gpu --iterations 10000 2>&1 |tee log/$term$delete_suffix.log
    #    let 'gpu=gpu+1'
done
