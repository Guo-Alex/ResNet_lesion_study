
import caffe
import operator as op

caffe.set_mode_cpu()
prototxt='CNN_bn.prototxt'

net=caffe.Net(prototxt, caffe.TEST)
params=[]
for pr in net.params:
    params.append(pr)
    conv_params=[pr for pr in params if pr.startswith('conv') and pr not in ['conv1', 'conv1_proj'] ]
summary=0
for pr in conv_params:
    factor = list(net.params[pr][0].data.shape)
    summary=summary+reduce(op.mul, factor)
    print "params {}: {},summary={}".format(pr, factor, summary)

print 'summary={}'.format(summary)
