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
from chainer import cuda
import chainer.functions as F
cuda.check_cuda_available()
xp = cuda.cupy 

###############################################################################
def feature_extraction(Xr, Xi, BPF, Log_reg):
    abs_X = F.sqrt( Xr**2 + Xi**2 + 1e-10 )         # amplitude 
    fet   = F.log( F.matmul(BPF, abs_X) + Log_reg ) # filterbank
    return fet

###############################################################################
def loss_MMSE(score):
    # MMSE between input and output
    loss = F.mean(score)
    return loss

#################################################################################
def return_time_mat_xp(x, fftl, shift):
    T = int( np.floor((len(x) - fftl)/float(shift)) )
    shift_matrix  = xp.tile( xp.arange( 0, shift*T, shift), (fftl, 1) )
    index_matrix  = xp.tile( xp.arange( 0, fftl ).reshape(fftl, 1), (1, T)) + shift_matrix
    return x[index_matrix.astype(np.int32)]

###############################################################################
def exe_fft(x, sp_param):
    # FFT
    X      = return_time_mat_xp(x, sp_param["fftl"], sp_param["shift"]).astype( np.float32 )
    O, T   = X.shape
    W      = xp.tile( sp_param["win"], (1, T) )
    Xr, Xi = chainer_fft_spectrogram( F.transpose(X*W), F.transpose(X*0), forward=True )
    return Xr, Xi

###############################################################################
def chainer_fft_spectrogram( xr, xi, forward=True):
    T, O = xr.shape
    if( forward ): # STFT
        yr, yi = F.fft( (xr,xi) )
        yr = F.transpose( yr[:, :int(O/2)+1] )
        yi = F.transpose( yi[:, :int(O/2)+1] )
    else:          # iSTFT
        xr_cnj = F.fliplr( xr[:,1:O-1] )
        xi_cnj = -F.fliplr( xi[:,1:O-1] )
        xr = F.concat( (xr,xr_cnj), axis=1 )
        xi = F.concat( (xi,xi_cnj), axis=1 )
        yr, yi = F.ifft( (xr,xi) )
        yr     = F.transpose( yr )
        yi     = F.transpose( yi )
    return yr, yi

###############################################################################
def concat_2_wavs(x, y, fftl):
    z = xp.hstack( (x, y) )
    return z

##############################################################################
if __name__ == "__main__":
     print('debug')   
    
    
    