import sys
import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
## net > block(skip and downsampling) > module

def add_module(bottom, num_output, pad, kernel_size, stride):
    conv = L.Convolution(bottom, param=[{'lr_mult':1, 'decay_mult':0}], num_output=num_output, 
                         pad=pad, kernel_size=kernel_size, stride=stride, bias_term=False,
                         weight_filler=dict(type='gaussian', std=round(0.01,2)))
    bn = L.BatchNorm(conv, moving_average_fraction=round(0.05,2), param=[{'lr_mult':0}, {'lr_mult':0},
                                                                {'lr_mult':0}], in_place=True)
    scale = L.Scale(conv, bias_term=True, in_place=True)
    return conv, bn, scale


def add_bottleneck(bottom, num_output, stride, flag):
    # flag: 1, skip; 2, downsampling
    # output: 11
    [conv1, bn1, scale1] = add_module(bottom, num_output/4, 0, 1, 1)
    relu1 = L.ReLURecover(conv1, in_place=True)
    [conv2, bn2, scale2] = add_module(conv1, num_output/4, 1, 3, 1)
    relu2 = L.ReLURecover(conv2, in_place=True)
    [conv3, bn3, scale3] = add_module(conv2, num_output*flag, 0, 1, stride)
    return conv1, bn1, scale1, relu1, conv2, bn2, scale2, relu2, conv3, bn3, scale3


def add_downsampling_block(bottom, num_output):
    # output: 16
    flag = 2
    [res1, bn1, scale1] = add_module(bottom, num_output*flag, 1, 3, 2)
    
    [conv1, bn2, scale2, relu1, conv2, bn3, scale3, relu2, conv3, bn4, scale4] = add_bottleneck(
        bottom, num_output, 2, flag=flag)
    res3 = L.Eltwise(res1, conv3)
    relu3 = L.ReLU(res3, in_place=True)
    return res1, bn1, scale1, conv1, bn2, scale2, relu1, conv2, bn3, scale3, relu2, \
    conv3, bn4, scale4, res3, relu3


def add_skip_block(bottom, num_output):
    # output: 13
    [conv1, bn1, scale1, relu1, conv2, bn2, scale2, relu2, conv3, bn3, scale3] = add_bottleneck(
        bottom, num_output, 1, flag=1)
    res = L.Eltwise(bottom, conv3)
    relu3 = L.ReLU(res, in_place=True)
    return conv1, bn1, scale1, relu1, conv2, bn2, scale2, relu2, conv3, bn3, scale3, res, relu3


def add_downsampling_block_1(bottom, num_output): # special case
    [res1, bn1, scale1] = add_module(bottom, 2*num_output, 1, 3, 2)
    [conv1, bn2, scale2] = add_module(bottom, num_output, 1, 3, 1)
    relu1 = L.ReLURecover(conv1, in_place=True)
    
    [conv2, bn3, scale3] = add_module(conv1, 2*num_output, 1, 3, 2)
    res2 = L.Eltwise(res1, conv2)
    relu2 = L.ReLU(res2, in_place=True)
    return res1, bn1, scale1, conv1, bn2, scale2, relu1, conv2, bn3, scale3, res2, relu2


def create_neural_net(input_file, batch_size=50):
    net = caffe.NetSpec()
    net.data, net.label = L.Data(batch_size=batch_size, source=input_file, 
                                  backend = caffe.params.Data.LMDB, ntop=2, 
                                  include=dict(phase=caffe.TEST), name='juniward04')

    ## pre-process
    net.conv1 = L.Convolution(net.data, num_output=16, kernel_size=4, stride=1,
                               pad=1, weight_filler=dict(type='dct4'),
                               param=[{'lr_mult':0, 'decay_mult':0}],
                               bias_term=False)
    TRUNCABS = caffe_pb2.QuantTruncAbsParameter.TRUNCABS
    net.quanttruncabs=L.QuantTruncAbs(net.conv1, process=TRUNCABS, threshold=8, in_place=True)

    ## block 1 16
    [net.conv1_proj, net.bn2, net.scale2, net.conv512_1, net.bn2_1, net.scale2_1,
     net.relu512_1, net.conv512_to_256, net.bn2_2, net.scale2_2, net.res512_to_256,
     net.relu512_to_256] = add_downsampling_block_1(net.quanttruncabs, 12)
    ## block 2 13
    [net.conv256_1, net.bn2_4, net.scale2_4, net.relu256_1, net.conv256_2, net.bn2_5, 
     net.scale2_5, net.relu256_2, net.conv256_3, net.bn2_6, net.scale2_6, net.res256_3, 
     net.relu256_3] = add_skip_block(net.res512_to_256, 24)
    ## block 3 16
    [net.res256_3_proj, net.bn2_7, net.scale2_7, net.conv256_4, net.bn2_8, net.scale2_8, 
     net.relu256_4, net.conv256_5, net.bn2_9, net.scale2_9, net.relu256_5, net.conv256_to_128, 
     net.bn2_10, net.scale2_10, net.res256_to_128, 
     net.relu256_to_128] = add_downsampling_block(net.res256_3, 24)
    ## block 4 13
    [net.conv128_1, net.bn2_11, net.scale2_11, net.relu128_1, net.conv128_2, net.bn2_12, 
     net.scale2_12, net.relu128_2, net.conv128_3, net.bn2_13, net.scale2_13,  net.res128_3, 
     net.relu128_3] = add_skip_block(net.res256_to_128, 48)
    ## block 5 16
    [net.res128_3_proj, net.bn2_14, net.scale2_14, net.conv128_4, net.bn2_15, net.scale2_15, 
     net.relu128_4, net.conv128_5, net.bn2_16, net.scale2_16, net.relu128_5, net.conv128_to_64, 
     net.bn2_17, net.scale2_17, net.res128_to_64, 
     net.relu128_to_64] = add_downsampling_block(net.res128_3, 48)
    ## block 6 13
    [net.conv64_1, net.bn2_18, net.scale2_18, net.relu64_1, net.conv64_2, net.bn2_19, 
     net.scale2_19, net.relu64_2, net.conv64_3, net.bn2_20, net.scale2_20, net.res64_3, 
     net.relu64_3] = add_skip_block(net.res128_to_64, 96)
    ## block 7 16
    [net.res64_3_proj, net.bn2_21, net.scale2_21, net.conv64_4, net.bn2_22, net.scale2_22, 
     net.relu64_4, net.con64_5, net.bn2_23, net.scale2_23, net.relu64_5, net.conv64_to_32, 
     net.bn2_24, net.scale2_24, net.res64_to_32, 
     net.relu64_to_32] = add_downsampling_block(net.res64_3, 96)
    ##    ## block 8 13
    ##    [net.conv32_1, net.bn2_25, net.scale2_25, net.relu32_1, net.conv32_2, net.bn2_26, 
    ##     net.scale2_26, net.relu32_2, net.conv32_3, net.bn2_27, net.scale2_27, net.res32_3, 
    ##     net.relu32_3] = add_skip_block(net.res64_to_32, 192)
    ## block 9 16
    [net.res32_3_proj, net.bn2_28, net.scale2_28, net.conv32_4, net.bn2_29, net.scale2_29, 
     net.relu32_4, net.con32_5, net.bn2_30, net.scale2_30, net.relu32_5, net.conv32_to_16, 
     net.bn2_31, net.scale2_31, net.res32_to_16, 
     net.relu32_to_16] = add_downsampling_block(net.res64_to_32, 192)
    ## block 10 13
    [net.conv16_1, net.bn2_32, net.scale2_32, net.relu16_1, net.conv16_2, net.bn2_33, 
     net.scale2_33, net.relu16_2, net.conv16_3, net.bn2_34, net.scale2_34, net.res16_3, 
     net.relu16_3] = add_skip_block(net.res32_to_16, 384)
    
    ## global pool
    AVE = caffe_pb2.PoolingParameter.AVE
    net.global_pool = L.Pooling(net.res16_3, pool=AVE, kernel_size=8, stride=1)
    
    ## full connecting
    net.fc = L.InnerProduct(net.global_pool, param=[{'lr_mult':1}, {'lr_mult':2}], num_output=2, 
                            weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    ## accuracy
    net.accuracy = L.Accuracy(net.fc, net.label, include=dict(phase=caffe.TEST))
    ## loss
    net.loss = L.SoftmaxWithLoss(net.fc, net.label)
    
    return net.to_proto()

if __name__=='__main__':
    train_file = sys.argv[1]
    output_file = sys.argv[2]
    # batch_size = 50
    with open(output_file, 'w') as f:
        f.write(str(create_neural_net(train_file)))
