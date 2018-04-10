#!/bin/sh

term=(1_4/ 9_12/ 17_20/)
for (( i=0; i<${#term[@]}; i=i+1 ))
do
    src='prototxt/'${term[i]}
    dst='caffemodel/'${term[i]}
    all_src=`ls $src |grep '.prototxt'`
    count=0
    #echo $count
    for file in $all_src
    do
        let 'count=count+1'
        src_file=$src$file
        delete_suffix=`echo $file |cut -d '.' -f 1`
        dst_file=$dst$delete_suffix'.caffemodel'
        echo $count': '$src_file'==>'$dst_file
        #    echo $count: $dst_file
        /usr/bin/python2.7 copy_some_parameter.py $src_file $dst_file
    done
done
