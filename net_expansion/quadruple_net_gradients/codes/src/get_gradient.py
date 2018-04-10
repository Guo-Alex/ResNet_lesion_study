
# lgq
# get the gradient when train mode
import caffe
from caffe.proto import caffe_pb2
import lmdb
import numpy as np

# deploy file
caffe_root='/home/lgq/Workspace/CIHW2018/net_expansion/quadruple_net_gradients/codes/'
caffe.set_mode_gpu()
solver=caffe.SGDSolver(caffe_root + 'src/solver.prototxt')
solver.restore(caffe_root+'src/snapshot/CNN_bn_50_iter_320000.solverstate')
net=solver.net
fc = net.params['fc'][0].data
np.save('fc.npy', fc)

solver.solve()

#out = net.forward()
fc_f = net.params['fc'][0].data
np.save('fc_f.npy', fc)

#diff = solver.net.backward()
#fc_b = net.params['fc'][0].data
#np.save('fc_b.npy', fc)
