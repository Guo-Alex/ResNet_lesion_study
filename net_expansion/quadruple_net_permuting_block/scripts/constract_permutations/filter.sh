#!/bin/sh
#lgq

src=$1
array=`cat $src |cut -d ' ' -f 21 | sort -u`
count=0
echo -n '' |tee filter

for factor in $array
do
    let 'count=count+1'
    echo -n $count': '
    temp=`cat $src |grep $factor |tail -1`
    echo $temp
    echo $temp >> filter
done
