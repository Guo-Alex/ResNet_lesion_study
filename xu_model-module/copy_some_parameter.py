
import caffe
import sys

caffe.set_mode_cpu()

old_caffemodel='caffemodel/CNN_bn_20.caffemodel'
new_model = sys.argv[1]
new_caffemodel = sys.argv[2]

print new_model
print new_caffemodel

net=caffe.Net(new_model,old_caffemodel,caffe.TEST)

#[(k, v[0].data.shape) for k, v in net.params.items()]
net.save(new_caffemodel)
