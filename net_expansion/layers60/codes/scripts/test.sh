#!/bin/sh
#$1:layers; $2:gpu id;

#term='module/downsampling/'
prototxt_file="CNN_bn_"$1".prototxt"
caffemodel="snapshot/"
all_caffemodel=`ls $caffemodel |grep caffemodel`
gpu=$2

for file in $all_caffemodel
do
    caffemodel_file=$caffemodel$file
    delete_suffix=`echo $file |cut -d '.' -f 1`
    log_file='../logs/test/2018.03/'$delete_suffix'.log'
    if [ ! -f $log_file ]; then
        echo $caffemodel_file' testing ...'
        sleep 1
        /home/lgq/Workspace/caffe/build/tools/caffe test --model $prototxt_file --weights $caffemodel_file --gpu $gpu --iterations 10000 2>&1 |tee $log_file
        #    let 'gpu=gpu+1'
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
