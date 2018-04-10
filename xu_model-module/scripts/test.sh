#!/bin/sh

#term='downsampling/'
term='skip/'
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
    
    log_file="log/$term$delete_suffix.log"
    if [ ! -f $log_file ]; then
        echo $caffemodel_file' testing ...'
        sleep 1
        /home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt_file --weights $caffemodel_file --gpu $gpu --iterations 10000 2>&1 |tee $log_file
    else
        error=`cat $log_file |cut -d ',' -f 1 |grep 'accuracy ='`
        if [ "$error" == "" ]; then
            rm $log_file
            echo $log_file' error, deleted'
            echo $caffemodel_file' testing ...'
            sleep 1
            /home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt_file --weights $caffemodel_file --gpu $gpu --iterations 10000 2>&1 |tee $log_file
        else
            echo $caffemodel_file'has been tested!'
            sleep 1
        fi
    fi
done
