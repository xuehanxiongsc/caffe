import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from net_util import max_pool
from net_util import conv_block
from net_util import eltwise_sum
from net_util import conv15x15_block
from net_util import conv31x31_block
from net_util import deconv_block

def fcn16(train_lmdb, val_lmdb, batch_size):
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
    pool1 = max_pool(conv1,2)
    conv2 = conv_block(pool1,1,64,stride=1,pad=0)
    conv3 = conv_block(conv1,2,16,stride=2,pad=0)
    conv4 = conv_block(conv3,3,16,stride=1,pad=1)
    conv5 = conv_block(conv4,1,64,stride=1,pad=0)
    sum0 = eltwise_sum(conv2,conv5)
    conv6 = conv_block(sum0,3,64,stride=1,pad=1)
    pool2 = max_pool(conv6,2)
    conv7 = conv15x15_block(pool2,128)
    fc0 = conv_block(conv7,1,128,stride=1,pad=0)
    fc1 = L.Convolution(fc0, kernel_size=1, num_output=4,stride=1,pad=0,
                        weight_filler=dict(type='xavier'))
    upsample = deconv_block(fc1,ks=16,nout=4,stride=8)
    crop = L.Crop(upsample,data,crop_param=dict(axis=2,offset=4))
    loss =  L.SoftmaxWithLoss(crop, label)
    acc = L.Accuracy(crop, label, accuracy_param=dict(top_k=1))
    return caffe.to_proto(loss,acc)
