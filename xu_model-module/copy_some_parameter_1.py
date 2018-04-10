
import caffe
import sys
caffe.set_mode_cpu()

old_caffemodel='caffemodel/CNN_bn.caffemodel'
old_model='prototxt/CNN_bn.prototxt'
#new_caffemodel='caffemodel/test.caffemodel'
new_model=sys.argv[1]
new_caffemodel=sys.argv[2]

net=caffe.Net(new_model,caffe.TEST)
net1=caffe.Net(old_model,old_caffemodel,caffe.TEST)

params=[]
for pr in net.params:
    params.append(pr)

#Let's transplant!
for pr in params:
    net.params[pr][0].data.flat = net1.params[pr][0].data.flat  # flat unrolls the arrays
    if pr.startswith('bn') or pr.startswith('scale') or pr=='fc':
        net.params[pr][1].data.flat = net1.params[pr][1].data.flat 
    if pr.startswith('bn'):
        net.params[pr][2].data.flat = net1.params[pr][2].data.flat 
        
    print "net params copy %s done!" % (pr)

#print 'Network initialize done!!'
net.save(new_caffemodel)

