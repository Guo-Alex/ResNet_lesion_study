#!/bin/sh
# lgq 
# extract accuracy from all the files in dictionary
# $1: dictionary

src=$1
all_src=`ls $src |grep '.log'`

for file in $all_src
do
    dst=`echo $file |cut -d '_' -f 5 |cut -d '.' -f 1`
    echo -n $dst
    accuracy=`cat $src$file |cut -d ',' -f 1 |grep 'accuracy = ' |cut -d '=' -f 2`
    echo ' '$accuracy
done
