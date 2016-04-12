caffe_root = '/home/kterao/sw/larbys_caffe' 

import sys,copy
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import tempfile

from caffe import layers as L
from caffe import params as P

def implement_alex(net,num_classes):
                    
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""

    weight_param  = dict(lr_mult=1, decay_mult=1)
    bias_param    = dict(lr_mult=2, decay_mult=0)
    learned_param = [weight_param, bias_param]

    try:
        net.data
        net.label
    except Exception:
        print 'data/label layer missing...'
        return False

    # 1st layer
    net.conv1 = L.Convolution(net.data, kernel_size=11, stride=4,
                              num_output=96, 
                              bias_filler=dict(type='constant', value=0),
                              weight_filler=dict(type='gaussian', std=0.01),
                              param=learned_param)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.norm1 = L.LRN(net.relu1, local_size=5, alpha=0.0001, beta=0.75)
    net.pool1 = L.Pooling(net.norm1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    # 2nd layer
    net.conv2 = L.Convolution(net.pool1, kernel_size=5, pad=2, group=2,
                              num_output=256, 
                              weight_filler=dict(type="gaussian",std=0.01),
                              bias_filler=dict(type="constant",value=0.1),
                              param=learned_param)
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.norm2 = L.LRN(net.relu2, local_size=5, alpha=0.0001, beta=0.75)
    net.pool2 = L.Pooling(net.norm2, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    # 3rd layer
    net.conv3 = L.Convolution(net.pool2, kernel_size=3, pad=1,
                              num_output=384,
                              weight_filler=dict(type="gaussian",std=0.01),
                              bias_filler=dict(type="constant",value=0),
                              param=learned_param)
    net.relu3 = L.ReLU(net.conv3, in_place=True)

    # 4th layer
    net.conv4 = L.Convolution(net.relu3, kernel_size=3, pad=1, group=2,
                              num_output=384,
                              weight_filler=dict(type="gaussian",std=0.01),
                              bias_filler=dict(type="constant",value=0.1),
                              param=learned_param)
    net.relu4 = L.ReLU(net.conv4, in_place=True)

    # 5th layer
    net.conv5 = L.Convolution(net.relu4, kernel_size=3, pad=1, group=2,
                              num_output=384,
                              weight_filler=dict(type="gaussian",std=0.01),
                              bias_filler=dict(type="constant",value=0.1),
                              param=learned_param)
    net.relu5 = L.ReLU(net.conv5, in_place=True)
    net.pool5 = L.Pooling(net.conv5, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    # 6th layer
    net.fc6 = L.InnerProduct(net.pool5, num_output=4096, param=learned_param,
                             weight_filler=dict(type="gaussian",std=0.005),
                             bias_filler=dict(type="constant",value=0.1))
    net.relu6 = L.ReLU(net.fc6, in_place=True)

    net.drop6 = L.Dropout(net.fc6, in_place=True, 
                          dropout_param=dict(dropout_ratio=0.5))

    # 7th layer
    net.fc7 = L.InnerProduct(net.fc6, num_output=4096, param=learned_param,
                             weight_filler=dict(type="gaussian",std=0.005),
                             bias_filler=dict(type="constant",value=0.1))
    net.relu7 = L.ReLU(net.fc7, in_place=True)
    net.drop7 = L.Dropout(net.fc7, in_place=True, 
                          dropout_param=dict(dropout_ratio=0.5))

    # 8th layer
    net.fc8 = L.InnerProduct( net.fc7,num_output=num_classes, param=learned_param,
                              weight_filler=dict(type="gaussian",std=0.01),
                              bias_filler=dict(type="constant",value=0))

    # Top layer
    net.accuracy = L.Accuracy(net.fc8, net.label,include=dict(phase=caffe.TEST))
    net.loss = L.SoftmaxWithLoss(net.fc8, net.label)
    return True


