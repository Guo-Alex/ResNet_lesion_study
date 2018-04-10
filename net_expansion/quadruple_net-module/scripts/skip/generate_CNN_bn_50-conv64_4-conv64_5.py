import sys
import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
## net > block(skip and downsampling) > module

def add_module(bottom, num_output, stride):
    conv = L.Convolution(bottom, param=[{'lr_mult':1, 'decay_mult':0}], num_output=num_output, 
                         pad=1, kernel_size=3, stride=stride, bias_term=False,
                         weight_filler=dict(type='gaussian', std=round(0.01,2)))
    bn = L.BatchNorm(conv, moving_average_fraction=round(0.05,2), param=[{'lr_mult':0}, {'lr_mult':0},
                                                                {'lr_mult':0}], in_place=True)
    scale = L.Scale(conv, bias_term=True, in_place=True)
    return conv, bn, scale


def add_downsampling_block(bottom, num_output):
    [res1, bn1, scale1] = add_module(bottom, 2*num_output, 2)
    [conv1, bn2, scale2] = add_module(bottom, num_output, 1)
    relu1 = L.ReLURecover(conv1, in_place=True)
    
    [conv2, bn3, scale3] = add_module(conv1, 2*num_output, 2)
    res2 = L.Eltwise(res1, conv2)
    relu2 = L.ReLU(res2, in_place=True)
    return res1, bn1, scale1, conv1, bn2, scale2, relu1, conv2, bn3, scale3, res2, relu2


def add_skip_block(bottom, num_output):
    [conv1, bn1, scale1] = add_module(bottom, num_output, 1)
    relu1 = L.ReLURecover(conv1, in_place=True)
    
    [conv2, bn2, scale2] = add_module(conv1, num_output, 1)
    res = L.Eltwise(bottom, conv2)
    relu2 = L.ReLU(res, in_place=True)
    return conv1, bn1, scale1, relu1, conv2, bn2, scale2, res, relu2


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

    ## block 1
    [net.conv1_proj, net.bn2, net.scale2, net.conv512_1, net.bn2_1, net.scale2_1,
     net.relu512_1, net.conv512_to_256, net.bn2_2, net.scale2_2, net.res512_to_256,
     net.relu512_to_256] = add_downsampling_block(net.quanttruncabs, 12)
    ## block 2
    [net.conv256_1, net.bn2_3, net.scale2_3, net.relu256_1, net.conv256_2, net.bn2_4, 
     net.scale2_4, net.res256_2, net.relu256_2] = add_skip_block(net.res512_to_256, 24)
    ## block 2_1
    [net.conv256_4, net.bn3_1, net.scale3_1, net.relu256_4, net.conv256_5, net.bn3_2, 
     net.scale3_2, net.res256_5, net.relu256_5] = add_skip_block(net.res256_2, 24)
    ## block 2_2
    [net.conv256_6, net.bn4_1, net.scale4_1, net.relu256_6, net.conv256_7, net.bn4_2, 
     net.scale4_2, net.res256_7, net.relu256_7] = add_skip_block(net.res256_5, 24)
    ## block 2_3
    [net.conv256_8, net.bn5_1, net.scale5_1, net.relu256_8, net.conv256_9, net.bn5_2, 
     net.scale5_2, net.res256_9, net.relu256_9] = add_skip_block(net.res256_7, 24)
    ## block 3
    [net.res256_2_proj, net.bn2_5, net.scale2_5, net.conv256_3, net.bn2_6, net.scale2_6, 
     net.relu256_3, net.conv256_to_128, net.bn2_7, net.scale2_7, net.res256_to_128, 
     net.relu256_to_128] = add_downsampling_block(net.res256_9, 24)
    ## block 4 
    [net.conv128_1, net.bn2_8, net.scale2_8, net.relu128_1, net.conv128_2, net.bn2_9, 
     net.scale2_9, net.res128_2, net.relu128_2] = add_skip_block(net.res256_to_128, 48)
    ## block 4_1
    [net.conv128_4, net.bn3_3, net.scale3_3, net.relu128_4, net.conv128_5, net.bn3_4, 
     net.scale3_4, net.res128_5, net.relu128_5] = add_skip_block(net.res128_2, 48)
    ## block 4_2
    [net.conv128_6, net.bn4_3, net.scale4_3, net.relu128_6, net.conv128_7, net.bn4_4, 
     net.scale4_4, net.res128_7, net.relu128_7] = add_skip_block(net.res128_5, 48)
    ## block 4_3
    [net.conv128_8, net.bn5_3, net.scale5_3, net.relu128_8, net.conv128_9, net.bn5_4, 
     net.scale5_4, net.res128_9, net.relu128_9] = add_skip_block(net.res128_7, 48)
    ## block 5
    [net.res128_2_proj, net.bn2_10, net.scale2_10, net.conv128_3, net.bn2_11, net.scale2_11, 
     net.relu128_3, net.conv128_to_64, net.bn2_12, net.scale2_12, net.res128_to_64, 
     net.relu128_to_64] = add_downsampling_block(net.res128_9, 48)
    ## block 6
    [net.conv64_1, net.bn2_13, net.scale2_13, net.relu64_1, net.conv64_2, net.bn2_14, 
     net.scale2_14, net.res64_2, net.relu64_2] = add_skip_block(net.res128_to_64, 96)
    ##    ## block 6_1
    ##    [net.conv64_4, net.bn3_5, net.scale3_5, net.relu64_4, net.conv64_5, net.bn3_6, 
    ##     net.scale3_6, net.res64_5, net.relu64_5] = add_skip_block(net.res64_2, 96)
    ## block 6_2
    [net.conv64_6, net.bn4_5, net.scale4_5, net.relu64_6, net.conv64_7, net.bn4_6, 
     net.scale4_6, net.res64_7, net.relu64_7] = add_skip_block(net.res64_2, 96)
    ## block 6_3
    [net.conv64_8, net.bn5_5, net.scale5_5, net.relu64_8, net.conv64_9, net.bn5_6, 
     net.scale5_6, net.res64_9, net.relu64_9] = add_skip_block(net.res64_7, 96)
    ## block 7
    [net.res64_2_proj, net.bn2_15, net.scale2_15, net.conv64_3, net.bn2_16, net.scale2_16, 
     net.relu64_3, net.conv64_to_32, net.bn2_17, net.scale2_17, net.res64_to_32, 
     net.relu64_to_32] = add_downsampling_block(net.res64_9, 96)
    ## block 8
    [net.conv32_1, net.bn2_18, net.scale2_18, net.relu32_1, net.conv32_2, net.bn2_19, 
     net.scale2_19, net.res32_2, net.relu32_2] = add_skip_block(net.res64_to_32, 192)
    ## block 8_1
    [net.conv32_4, net.bn3_7, net.scale3_7, net.relu32_4, net.conv32_5, net.bn3_8, 
     net.scale3_8, net.res32_5, net.relu32_5] = add_skip_block(net.res32_2, 192)
    ## block 8_2
    [net.conv32_6, net.bn4_7, net.scale4_7, net.relu32_6, net.conv32_7, net.bn4_8, 
     net.scale4_8, net.res32_7, net.relu32_7] = add_skip_block(net.res32_5, 192)
    ## block 8_3
    [net.conv32_8, net.bn5_7, net.scale5_7, net.relu32_8, net.conv32_9, net.bn5_8, 
     net.scale5_8, net.res32_9, net.relu32_9] = add_skip_block(net.res32_7, 192)
    ## block 9
    [net.res32_2_proj, net.bn2_20, net.scale2_20, net.conv32_3, net.bn2_21, net.scale2_21, 
     net.relu32_3, net.conv32_to_16, net.bn2_22, net.scale2_22, net.res32_to_16, 
     net.relu32_to_16] = add_downsampling_block(net.res32_9, 192)
    ## block 10
    [net.conv16_1, net.bn2_23, net.scale2_23, net.relu16_1, net.conv16_2, net.bn2_24, 
     net.scale2_24, net.res16_2, net.relu16_2] = add_skip_block(net.res32_to_16, 384)
    ## block 10_1
    [net.conv16_3, net.bn3_9, net.scale3_9, net.relu16_3, net.conv16_4, net.bn3_10, 
     net.scale3_10, net.res16_4, net.relu16_4] = add_skip_block(net.res16_2, 384)
    ## block 10_2
    [net.conv16_5, net.bn4_9, net.scale4_9, net.relu16_5, net.conv16_6, net.bn4_10, 
     net.scale4_10, net.res16_6, net.relu16_6] = add_skip_block(net.res16_4, 384)
    ## block 10_3
    [net.conv16_7, net.bn5_9, net.scale5_9, net.relu16_7, net.conv16_8, net.bn5_10, 
     net.scale5_10, net.res16_8, net.relu16_8] = add_skip_block(net.res16_6, 384)
    
    ## global pool
    AVE = caffe_pb2.PoolingParameter.AVE
    net.global_pool = L.Pooling(net.res16_8, pool=AVE, kernel_size=8, stride=1)
    
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
