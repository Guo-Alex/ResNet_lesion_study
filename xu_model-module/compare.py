
import caffe
caffe.set_mode_cpu()

caffemodel1='caffemodel/CNN_bn-conv32_1.caffemodel'
model='prototxt/CNN_bn-conv32_1.prototxt'
caffemodel2='caffemodel/copy_some_parameter/CNN_bn-conv32_1.caffemodel'

net1=caffe.Net(model, caffemodel1, caffe.TEST)
net2=caffe.Net(model, caffemodel2, caffe.TEST)

params=[]
for pr in net2.params:
    params.append(pr)

flag=True
for pr in params:
    flag=(net1.params[pr][0].data==net2.params[pr][0].data).all()
    if(not flag):
        print 'params[{}][0]:False'.format(pr)
        break
    if pr.startswith('bn') or pr.startswith('scale') or pr=='fc':
        flag = (net1.params[pr][1].data==net2.params[pr][1].data).all()
        if(not flag):
            print 'params[{}][1]:False'.format(pr)
            break
        
if(flag):
    print 'True'
