# -*- coding: utf-8 -*-
"""
PyTorch-compatible config loader.
Reuses the original Config.py parameters (no Chainer dependency).
"""
import sys
import os

# Import the original config (pure Python dicts, no framework dependency)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Modules'))
import Config as _Config


def load_config():
    """Load signal processing, DNN, and training parameters."""
    sp_param, dnn_param, training_param = _Config.load_config()
    return sp_param, dnn_param, training_param
