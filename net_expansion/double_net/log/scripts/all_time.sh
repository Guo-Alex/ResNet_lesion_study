#!/bin/sh

cat xu_model_retrain_batch40.log |grep ^I0 |cut -d ':' -f 1 |uniq
