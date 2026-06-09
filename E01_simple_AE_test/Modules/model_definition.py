# -*- coding: utf-8 -*-
import numpy as np
import sys
import types
# Monkey-patch np.sctypes for NumPy 2.0+ compatibility (required by chainer)
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'float':   [np.float16, np.float32, np.float64],
        'int':     [np.int8, np.int16, np.int32, np.int64],
        'uint':    [np.uint8, np.uint16, np.uint32, np.uint64],
        'complex': [np.complex64, np.complex128],
        'others':  [bool, object, bytes, str, np.void],
    }
# Stub numpy.distutils for NumPy 2.0+ (removed in 2.0, but scipy may still import it)
if 'numpy.distutils' not in sys.modules:
    sys.modules['numpy.distutils'] = types.ModuleType('numpy.distutils')
    for _sub in ('misc_util', 'log', 'exec_command', 'core', 'ccompiler',
                  'fcompiler', 'system_info', 'cpuinfo', 'line_endings',
                  'npy_pkg_config', 'pathccompiler', 'unixccompiler',
                  'msvc9compiler', 'mingw32ccompiler', 'intelccompiler',
                  'conv_template', 'from_template', 'extension',
                  'command', 'armccompiler'):
        _full = f'numpy.distutils.{_sub}'
        if _full not in sys.modules:
            sys.modules[_full] = types.ModuleType(_full)
import chainer
from chainer import cuda, Chain
import chainer.functions as F
import chainer.links as L
cuda.check_cuda_available()
xp          = cuda.cupy 

################################################################
def frame_concat(h, bw, fw):
    p    = h
    K, O = h.shape
    for ii in range( bw ):
        z = xp.zeros( (ii, O), dtype=np.float32 )
        q = F.concat( (p[ii:,:],z), axis=0)
        h = F.concat( (h,q), axis=1 )
    for ii in range( fw ):
        z = xp.zeros( (ii+1, O), dtype=np.float32 )
        q = F.concat( (z, p[0:K-(ii+1),:]), axis=0)
        h = F.concat( (h,q), axis=1 )
    return h
    
###############################################################################
class FCN_AE(Chain):
    # NNの構造を記述
    def __init__(self, 
                 in_dim, 
                 hid_dim, 
                 z_dim, 
                 num_hid, 
                 num_fw,
                 num_bw):
        super(FCN_AE, self).__init__()
        with self.init_scope():
            self.hidden_layer_num = num_hid    
            self.num_fw           = num_fw
            self.num_bw           = num_bw
            initializer         = chainer.initializers.GlorotNormal()
            ###################################################################
            # Input BN
            self.in_BN = L.BatchNormalization( in_dim, use_gamma=False, use_beta=False )
            ###################################################################    
            # Encoder            
            self.add_layer('e_in', L.Linear( in_dim*(1+self.num_fw+self.num_bw), 
                                            hid_dim, initialW=initializer))
            for i in range(1, 1+self.hidden_layer_num):
                self.add_layer( 'e_%d' % i, L.Linear( hid_dim, hid_dim, initialW=initializer))
            self.add_layer('e_out', L.Linear(hid_dim, z_dim, initialW=initializer))
            ###################################################################    
            # Decoder            
            self.add_layer('d_in', L.Linear( z_dim, hid_dim, initialW=initializer))
            for i in range(1, 1+self.hidden_layer_num):
                self.add_layer( 'd_%d' % i, L.Linear( hid_dim, hid_dim, initialW=initializer))
            self.add_layer('d_out', L.Linear(hid_dim, 
                                             in_dim*(1+self.num_fw+self.num_bw), initialW=initializer))
            ###################################################################
            # Show number of params
            print('Number of parameters: '+str( sum(p.data.size for p in self.params()) ))
    # Add link
    def add_layer(self, name, function):
        super(FCN_AE, self).add_link(name, function)
    # input normalization using BN
    def input_bn(self, x_data):
        x = self.in_BN( F.transpose( x_data ) )
        return x
    # forward 
    def __call__(self, x_data):
        #######################################################################
        # Input normalization using batch-normalization
        x   = self.input_bn(x_data)
        x   = frame_concat(x, self.num_fw, self.num_bw)
        h   = x
        #######################################################################
        # Encoder
        h = F.relu( self.e_in(h) ) 
        for i in range(1, 1+self.hidden_layer_num):
            h   = F.relu( getattr(self,  'e_%d' % i)(h) )
        z = F.relu( self.e_out(h) ) 
        #######################################################################
        # Decoder
        h = F.relu( self.d_in(z) ) 
        for i in range(1, 1+self.hidden_layer_num):
            h   = F.relu( getattr(self,  'd_%d' % i)(h) )
        y = self.d_out(h) 
        #######################################################################
        # score
        score = F.sum( (x-y)**2, axis=1 ) / (1+self.num_fw+self.num_bw)
        return score, x, y   
################################################################


    
    
    
    



