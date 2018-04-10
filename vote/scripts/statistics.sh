#!/bin/sh

log='logs/'
all_logs=`ls $log |grep CNN_bn`
dst=$log'statistics.txt'
if [ -f $dst ]; then
    rm -f $dst
fi
for file in $all_logs
do
    cat $log$file |grep 'Batch ' |grep 'accuracy ' |cut -d '=' -f 2 > temp
    #echo -n $file': ' >> $dst
    awk '{for(i=1;i<=NF;i=i+1){a[NR,i]=$i}}END{for(j=1;j<=NF;j++){str=a[1,j];for(i=2;i<=NR;i++){str=str " " a[i,j]}print str}}' temp >> $dst
done
rm -f temp
