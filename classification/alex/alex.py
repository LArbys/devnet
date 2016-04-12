from alex_net import implement_alex

caffe_root = '/home/kterao/sw/larbys_caffe' 
import sys,copy
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()

import numpy as np
import tempfile

from caffe import layers as L
from caffe import params as P

weight_param  = dict(lr_mult=1, decay_mult=1)
bias_param    = dict(lr_mult=2, decay_mult=0)
lrn_param     = dict(local_size=5,alpha=0.0001,beta=0.75)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

TRAIN_DATA='/mnt/disk0/kterao/larbys/singlep_lmdb/singlep_train_v5.db'
TRAIN_DATA_MEAN='/mnt/disk0/kterao/larbys/singlep_lmdb/singlep_train_v5.db.bin'
VALIDATION_DATA='/mnt/disk0/kterao/larbys/singlep_lmdb/singlep_validate_v5.db'
VALIDATION_DATA_MEAN='/mnt/disk0/kterao/larbys/singlep_lmdb/singlep_validate_v5.db.bin'

def brew_alex(train=False):

    net = caffe.NetSpec()
    if train:
        net.data, net.label = L.Data( transform_param=dict(mirror=True,
                                                           crop_size=224,
                                                           mean_file=TRAIN_DATA_MEAN),
                                      source=TRAIN_DATA,
                                      batch_size=128, 
                                      backend=1,
                                      ntop=2 )
    else:
        net.data, net.label = L.Data( transform_param=dict(mirror=True,
                                                           crop_size=224,
                                                           mean_file=VALIDATION_DATA_MEAN),
                                      source=VALIDATION_DATA,
                                      batch_size=128, 
                                      backend=1,
                                      ntop=2 )

    implement_alex(net,5)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(net.to_proto()))
        return f.name

def solver(snapshot_prefix, train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe.proto.caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = snapshot_prefix
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

if __name__ == '__main__':

    train_net = brew_alex(True)
    test_net  = brew_alex(False)
    solver_cfg = solver('snapshot',train_net,test_net)

    solver = caffe.SGDSolver(solver_cfg)
    solver.solve()
    solver.net.save('snapshot_final')
