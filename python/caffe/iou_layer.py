import caffe
import numpy as np

class IOULayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two layer.bottoms'
        assert len(top) == 1,    'requires a single layer.top'

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
        top[0].data[...] = bottom[0].data + self.num

    def backward(self, top, propagate_down, bottom):
        pass
