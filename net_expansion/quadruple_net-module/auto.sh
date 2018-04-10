#!/bin/sh

src='prototxt/downsampling/'
dst='caffemodel/downsampling/'
#src='prototxt/skip/'
#dst='caffemodel/skip/'
all_src=`ls $src |grep '-'`
count=0

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
