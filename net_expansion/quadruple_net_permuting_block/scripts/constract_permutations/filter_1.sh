#!/bin/sh
#lgq
#$1 begin
#$2 end

echo -e $1'\t'$2
src='coeff'
array=`cat $src |cut -d ' ' -f 21 | sort -u`
count=0

for factor in $array
do
    let 'count=count+1'
    echo -n $count': '
    temp=`cat $src |grep $factor |grep "$1" |grep "$2" |tail -1`
    echo $temp
done
