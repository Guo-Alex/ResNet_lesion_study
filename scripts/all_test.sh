#!/bin/sh
#$1 gpu number
root='/home/lgq/Workspace/CIHW2018/'
xu_20='xu_model'
bottle_neck='bottleneck'
xu_30='net_expansion/double_net'
xu_40='net_expansion/triple_net'
xu_50='net_expansion/quadruple_net'
#array=($xu_20 $bottle_neck $xu_30 $xu_40 $xu_50)
array=($xu_50 $xu_40 $xu_30 $xu_20 $bottle_neck)
for (( i=0; i<${#array[@]}; i=i+1 ))
do
    cd $root${array[i]}'/codes/'
    pwd
    sh scripts/test.sh $1
    echo ${array[i]}' done'
done
cd $root
