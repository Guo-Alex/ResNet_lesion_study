#!/bin/sh

src=$1

for (( i=256; i>=16; i=i/2 ))
do
    if [ "$1" == "downsampling/" ]; then
        let 'flag=i*2'
    elif [ "$1" == "skip/" ]; then
        let 'flag=i'
    else
        echo 'first parameter error'
        exit 1
    fi
    all_src=`ls $src |grep $flag`

    for file in $all_src
    do
        #        echo $file
        accuracy=`cat $src$file |cut -d ',' -f 1 |grep 'accuracy = ' |cut -d '=' -f 2`
        echo $accuracy
    done
done
