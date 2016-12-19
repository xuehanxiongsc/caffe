import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from net_util import max_pool
from net_util import conv_block
from net_util import eltwise_sum
from net_util import conv15x15_block
from net_util import conv31x31_block
from net_util import deconv_block
from net_util import bottleneck3x3_block
from net_util import bottleneck5x5_block
from net_util import bottleneck9x9_block
from net_util import bottleneck2_block

def fcn16_enet(train_lmdb, val_lmdb, batch_size):
    data, label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, 
                         source=train_lmdb,
                         transform_param=dict(crop_size=128,mirror=True), 
                         ntop=2,
                         include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, 
                         source=val_lmdb,
                         transform_param=dict(crop_size=128), 
                         ntop=2,
                         include=dict(phase=getattr(caffe_pb2, 'TEST')))
    conv1 = conv_block(data,3,16,stride=2,pad=1)
    sum0 = bottleneck2_block(conv1,16)
    sum1 = bottleneck2_block(sum0, 32) 
    bn0 = bottleneck3x3_block(sum1,32)
    bn1 = bottleneck3x3_block(bn0,32,dilation=2)
    bn2 = bottleneck5x5_block(bn1,32)
    bn3 = bottleneck3x3_block(bn2,32,dilation=4)
    fc0 = L.Convolution(bn3, kernel_size=1, num_output=4,stride=1,pad=0,
                        weight_filler=dict(type='xavier'))
    upsample = deconv_block(fc0,ks=16,nout=4,stride=8)
    crop = L.Crop(upsample,data,crop_param=dict(axis=2,offset=4))
    loss =  L.SoftmaxWithLoss(crop, label)
    acc = L.Accuracy(crop, label, accuracy_param=dict(top_k=1))
    return caffe.to_proto(loss,acc)
