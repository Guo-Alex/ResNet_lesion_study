#!/usr/bin/python2.7
import caffe
import sys

caffe.set_mode_cpu()

old_caffemodel='caffemodel/CNN_bn_bottleneck.caffemodel'
# new_model = 'prototxt/skip/CNN_bn_40-conv128_1-conv128_2.prototxt'
# new_caffemodel = 'caffemodel/skip/CNN_bn_40-conv128_1-conv128_2.caffemodel'
new_model = sys.argv[1]
new_caffemodel = sys.argv[2]

print(new_model)
print(new_caffemodel)

net=caffe.Net(new_model,old_caffemodel,caffe.TEST)

#[(k, v[0].data.shape) for k, v in net.params.items()]
net.save(new_caffemodel)
