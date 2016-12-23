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

def conv_rect_block(bottom,nout,kernel_h=3,kernel_w=3,stride=1,pad_h=0,pad_w=0):
    conv = L.Convolution(bottom, kernel_w=kernel_w, kernel_h=kernel_h,
                         num_output=nout,stride=stride,pad_w=pad_w,pad_h=pad_h,
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
                               bias_term=False,
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

def bottleneck3x3_block(bottom,nin,dilation=1):
    conv = conv_block(bottom,1,nin,pad=0)
    conv = conv_block(conv,3,nin,pad=dilation,dilation=dilation)
    conv = conv_block(conv,1,nin*4,pad=0)
    top = eltwise_sum(bottom,conv)
    return top 

def bottleneck5x5_block(bottom,nin,dilation=1):
    conv = conv_block(bottom,1,nin,pad=0)
    conv = conv_rect_block(conv,nin,kernel_w=5,kernel_h=1,
                           pad_w=2,pad_h=0)
    conv = conv_rect_block(conv,nin,kernel_w=1,kernel_h=5,
                           pad_w=0,pad_h=2)
    conv = conv_block(conv,1,nin*4,pad=0)
    top = eltwise_sum(bottom,conv)
    return top 

def bottleneck9x9_block(bottom,nin,dilation=1):
    conv = conv_block(bottom,1,nin,pad=0)
    conv = conv_rect_block(conv,nin,kernel_w=9,kernel_h=1,
                           pad_w=4,pad_h=0)
    conv = conv_rect_block(conv,nin,kernel_w=1,kernel_h=9,
                           pad_w=0,pad_h=4)
    conv = conv_block(conv,1,nin*4,pad=0)
    top = eltwise_sum(bottom,conv)
    return top 

def bottleneck2_block(bottom,nin):
    pool1 = max_pool(bottom,2)
    conv2 = conv_block(pool1,1,nin*4,stride=1,pad=0)
    conv3 = conv_block(bottom,2,nin,stride=2,pad=0)
    conv4 = conv_block(conv3,3,nin,stride=1,pad=1)
    conv5 = conv_block(conv4,1,nin*4,stride=1,pad=0)
    sum0 = eltwise_sum(conv2,conv5)
    return sum0

