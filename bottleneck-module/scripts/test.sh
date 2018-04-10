#!/bin/sh

term=(downsampling/ skip/)
for (( i=0; i<${#term[@]}; i=i+1 ))
do
    prototxt="prototxt/${term[i]}"
    caffemodel="caffemodel/${term[i]}"
    all_prototxt=`ls $prototxt`
    gpu=0

    for file in $all_prototxt
    do
        #    let 'gpu=count/4'
        prototxt_file=$prototxt$file
        delete_suffix=`echo $file |cut -d '.' -f 1`
        caffemodel_file=$caffemodel$delete_suffix'.caffemodel'
        
        log_file="log/${term[i]}$delete_suffix.log"
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
done
