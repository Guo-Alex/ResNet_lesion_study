#!/bin/sh

test_file='/data/lgq/basic500k/lmdb/test'
src=(skip/ downsampling/)
for (( i=0; i<${#src[@]}; i=i+1 ))
do
    echo ${src[i]}
    dst='../prototxt/'${src[i]}
    
    all_src=`ls ${src[i]} |grep '.py'`
    count=0
    
    for file in $all_src
    do
        let 'count=count+1'
        src_file=${src[i]}$file
        name=`echo $file |cut -d '_' -f 2-6 |cut -d '.' -f 1`
        dst_file=$dst$name'.prototxt'
        echo $count': generate '$dst_file
        python $src_file $test_file $dst_file
    done
done
