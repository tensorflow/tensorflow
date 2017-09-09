# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras backend API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=redefined-builtin
from tensorflow.python.keras._impl.keras.backend import abs
from tensorflow.python.keras._impl.keras.backend import all
from tensorflow.python.keras._impl.keras.backend import any
from tensorflow.python.keras._impl.keras.backend import arange
from tensorflow.python.keras._impl.keras.backend import argmax
from tensorflow.python.keras._impl.keras.backend import argmin
from tensorflow.python.keras._impl.keras.backend import backend
from tensorflow.python.keras._impl.keras.backend import batch_dot
from tensorflow.python.keras._impl.keras.backend import batch_flatten
from tensorflow.python.keras._impl.keras.backend import batch_get_value
from tensorflow.python.keras._impl.keras.backend import batch_normalization
from tensorflow.python.keras._impl.keras.backend import batch_set_value
from tensorflow.python.keras._impl.keras.backend import bias_add
from tensorflow.python.keras._impl.keras.backend import binary_crossentropy
from tensorflow.python.keras._impl.keras.backend import cast
from tensorflow.python.keras._impl.keras.backend import cast_to_floatx
from tensorflow.python.keras._impl.keras.backend import categorical_crossentropy
from tensorflow.python.keras._impl.keras.backend import clear_session
from tensorflow.python.keras._impl.keras.backend import clip
from tensorflow.python.keras._impl.keras.backend import concatenate
from tensorflow.python.keras._impl.keras.backend import constant
from tensorflow.python.keras._impl.keras.backend import conv1d
from tensorflow.python.keras._impl.keras.backend import conv2d
from tensorflow.python.keras._impl.keras.backend import conv2d_transpose
from tensorflow.python.keras._impl.keras.backend import conv3d
from tensorflow.python.keras._impl.keras.backend import cos
from tensorflow.python.keras._impl.keras.backend import count_params
from tensorflow.python.keras._impl.keras.backend import ctc_batch_cost
from tensorflow.python.keras._impl.keras.backend import ctc_decode
from tensorflow.python.keras._impl.keras.backend import ctc_label_dense_to_sparse
from tensorflow.python.keras._impl.keras.backend import dot
from tensorflow.python.keras._impl.keras.backend import dropout
from tensorflow.python.keras._impl.keras.backend import dtype
from tensorflow.python.keras._impl.keras.backend import elu
from tensorflow.python.keras._impl.keras.backend import epsilon
from tensorflow.python.keras._impl.keras.backend import equal
from tensorflow.python.keras._impl.keras.backend import eval
from tensorflow.python.keras._impl.keras.backend import exp
from tensorflow.python.keras._impl.keras.backend import expand_dims
from tensorflow.python.keras._impl.keras.backend import eye
from tensorflow.python.keras._impl.keras.backend import flatten
from tensorflow.python.keras._impl.keras.backend import floatx
from tensorflow.python.keras._impl.keras.backend import foldl
from tensorflow.python.keras._impl.keras.backend import foldr
from tensorflow.python.keras._impl.keras.backend import function
from tensorflow.python.keras._impl.keras.backend import gather
from tensorflow.python.keras._impl.keras.backend import get_session
from tensorflow.python.keras._impl.keras.backend import get_uid
from tensorflow.python.keras._impl.keras.backend import get_value
from tensorflow.python.keras._impl.keras.backend import gradients
from tensorflow.python.keras._impl.keras.backend import greater
from tensorflow.python.keras._impl.keras.backend import greater_equal
from tensorflow.python.keras._impl.keras.backend import hard_sigmoid
from tensorflow.python.keras._impl.keras.backend import image_data_format
from tensorflow.python.keras._impl.keras.backend import in_test_phase
from tensorflow.python.keras._impl.keras.backend import in_top_k
from tensorflow.python.keras._impl.keras.backend import in_train_phase
from tensorflow.python.keras._impl.keras.backend import int_shape
from tensorflow.python.keras._impl.keras.backend import is_sparse
from tensorflow.python.keras._impl.keras.backend import l2_normalize
from tensorflow.python.keras._impl.keras.backend import learning_phase
from tensorflow.python.keras._impl.keras.backend import less
from tensorflow.python.keras._impl.keras.backend import less_equal
from tensorflow.python.keras._impl.keras.backend import log
from tensorflow.python.keras._impl.keras.backend import manual_variable_initialization
from tensorflow.python.keras._impl.keras.backend import map_fn
from tensorflow.python.keras._impl.keras.backend import max
from tensorflow.python.keras._impl.keras.backend import maximum
from tensorflow.python.keras._impl.keras.backend import mean
from tensorflow.python.keras._impl.keras.backend import min
from tensorflow.python.keras._impl.keras.backend import minimum
from tensorflow.python.keras._impl.keras.backend import moving_average_update
from tensorflow.python.keras._impl.keras.backend import name_scope
from tensorflow.python.keras._impl.keras.backend import ndim
from tensorflow.python.keras._impl.keras.backend import normalize_batch_in_training
from tensorflow.python.keras._impl.keras.backend import not_equal
from tensorflow.python.keras._impl.keras.backend import one_hot
from tensorflow.python.keras._impl.keras.backend import ones
from tensorflow.python.keras._impl.keras.backend import ones_like
from tensorflow.python.keras._impl.keras.backend import permute_dimensions
from tensorflow.python.keras._impl.keras.backend import placeholder
from tensorflow.python.keras._impl.keras.backend import pool2d
from tensorflow.python.keras._impl.keras.backend import pool3d
from tensorflow.python.keras._impl.keras.backend import pow
from tensorflow.python.keras._impl.keras.backend import print_tensor
from tensorflow.python.keras._impl.keras.backend import prod
from tensorflow.python.keras._impl.keras.backend import random_binomial
from tensorflow.python.keras._impl.keras.backend import random_normal
from tensorflow.python.keras._impl.keras.backend import random_normal_variable
from tensorflow.python.keras._impl.keras.backend import random_uniform
from tensorflow.python.keras._impl.keras.backend import random_uniform_variable
from tensorflow.python.keras._impl.keras.backend import relu
from tensorflow.python.keras._impl.keras.backend import repeat
from tensorflow.python.keras._impl.keras.backend import repeat_elements
from tensorflow.python.keras._impl.keras.backend import reset_uids
from tensorflow.python.keras._impl.keras.backend import reshape
from tensorflow.python.keras._impl.keras.backend import resize_images
from tensorflow.python.keras._impl.keras.backend import resize_volumes
from tensorflow.python.keras._impl.keras.backend import reverse
from tensorflow.python.keras._impl.keras.backend import rnn
from tensorflow.python.keras._impl.keras.backend import round
from tensorflow.python.keras._impl.keras.backend import separable_conv2d
from tensorflow.python.keras._impl.keras.backend import set_epsilon
from tensorflow.python.keras._impl.keras.backend import set_floatx
from tensorflow.python.keras._impl.keras.backend import set_image_data_format
from tensorflow.python.keras._impl.keras.backend import set_learning_phase
from tensorflow.python.keras._impl.keras.backend import set_session
from tensorflow.python.keras._impl.keras.backend import set_value
from tensorflow.python.keras._impl.keras.backend import shape
from tensorflow.python.keras._impl.keras.backend import sigmoid
from tensorflow.python.keras._impl.keras.backend import sign
from tensorflow.python.keras._impl.keras.backend import sin
from tensorflow.python.keras._impl.keras.backend import softmax
from tensorflow.python.keras._impl.keras.backend import softplus
from tensorflow.python.keras._impl.keras.backend import softsign
from tensorflow.python.keras._impl.keras.backend import sparse_categorical_crossentropy
from tensorflow.python.keras._impl.keras.backend import spatial_2d_padding
from tensorflow.python.keras._impl.keras.backend import spatial_3d_padding
from tensorflow.python.keras._impl.keras.backend import sqrt
from tensorflow.python.keras._impl.keras.backend import square
from tensorflow.python.keras._impl.keras.backend import squeeze
from tensorflow.python.keras._impl.keras.backend import stack
from tensorflow.python.keras._impl.keras.backend import std
from tensorflow.python.keras._impl.keras.backend import stop_gradient
from tensorflow.python.keras._impl.keras.backend import sum
from tensorflow.python.keras._impl.keras.backend import switch
from tensorflow.python.keras._impl.keras.backend import tanh
from tensorflow.python.keras._impl.keras.backend import temporal_padding
from tensorflow.python.keras._impl.keras.backend import to_dense
from tensorflow.python.keras._impl.keras.backend import transpose
from tensorflow.python.keras._impl.keras.backend import truncated_normal
from tensorflow.python.keras._impl.keras.backend import update
from tensorflow.python.keras._impl.keras.backend import update_add
from tensorflow.python.keras._impl.keras.backend import update_sub
from tensorflow.python.keras._impl.keras.backend import var
from tensorflow.python.keras._impl.keras.backend import variable
from tensorflow.python.keras._impl.keras.backend import zeros
from tensorflow.python.keras._impl.keras.backend import zeros_like

del absolute_import
del division
del print_function
