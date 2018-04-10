#!/bin/sh
# lgq 
# $1: file

src=$1
cat $src |grep 'Train net' | grep 'loss = ' |cut -d '=' -f 2 |cut -d '(' -f 1 |tee loss
