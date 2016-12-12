from caffe import layers as L, params as P

def eltwise_sum(bottom0,bottom1):
    eltwise_sum = L.Eltwise(bottom0, bottom1, operation=P.Eltwise.SUM)
    return eltwise_sum

def max_pool(bottom, ks, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv_block(bottom,ks,nout,stride=1,pad=0,dilation=1):
    conv = L.Convolution(bottom, kernel_size=ks, num_output=nout, 
                         stride=stride, pad=pad,dilation=dilation,
                         weight_filler=dict(type='xavier'))
    batch_norm = L.BatchNorm(conv, in_place=True, 
                             param=[dict(lr_mult=0, decay_mult=0), 
                                    dict(lr_mult=0, decay_mult=0), 
                                    dict(lr_mult=0, decay_mult=0)])
    relu = L.ReLU(batch_norm, in_place=True)
    return relu

def deconv_block(bottom,ks,nout,stride,pad=0):
    deconv = L.Deconvolution(
        bottom,                      
        convolution_param=dict(num_output=nout, 
                               kernel_size=ks, 
                               stride=stride,
                               pad=pad,
                               weight_filler=dict(type='xavier')))
    return deconv

def conv15x15_block(bottom,nout):
    conv = conv_block(bottom,3,nout,pad=1)
    conv = conv_block(conv,3,nout,pad=2,dilation=2)
    conv = conv_block(conv,3,nout,pad=4,dilation=4)
    return conv

def conv31x31_block(bottom,nout):
    conv = conv_block(bottom,3,nout,pad=1)
    conv = conv_block(conv,3,nout,pad=2,dilation=2)
    conv = conv_block(conv,3,nout,pad=4,dilation=4)
    conv = conv_block(conv,3,nout,pad=8,dilation=8)
    return conv

