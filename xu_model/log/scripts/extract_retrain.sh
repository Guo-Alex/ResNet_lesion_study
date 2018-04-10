#!/bin/sh
#cat xu_model_retrain.log |grep 'accuracy =' |cut -d '=' -f 2 |tee temp
#cat xu_model_retrain_next.log |grep 'accuracy =' |cut -d '=' -f 2
#cat xu_model_retrain_next.log |grep 'accuracy =' |cut -d '=' -f 2 >> temp
#cat xu_model_retrain_batch40.log |grep 'accuracy =' |cut -d '=' -f 2 |tee temp
#cat xu_model_retrain_stepsize12500.log |grep 'accuracy =' |cut -d '=' -f 2 |tee temp
cat $1 |grep 'accuracy =' |cut -d '=' -f 2 |tee temp
