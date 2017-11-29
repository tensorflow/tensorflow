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
from tensorflow.contrib.keras.python.keras.backend import abs
from tensorflow.contrib.keras.python.keras.backend import all
from tensorflow.contrib.keras.python.keras.backend import any
from tensorflow.contrib.keras.python.keras.backend import arange
from tensorflow.contrib.keras.python.keras.backend import argmax
from tensorflow.contrib.keras.python.keras.backend import argmin
from tensorflow.contrib.keras.python.keras.backend import backend
from tensorflow.contrib.keras.python.keras.backend import batch_dot
from tensorflow.contrib.keras.python.keras.backend import batch_flatten
from tensorflow.contrib.keras.python.keras.backend import batch_get_value
from tensorflow.contrib.keras.python.keras.backend import batch_normalization
from tensorflow.contrib.keras.python.keras.backend import batch_set_value
from tensorflow.contrib.keras.python.keras.backend import bias_add
from tensorflow.contrib.keras.python.keras.backend import binary_crossentropy
from tensorflow.contrib.keras.python.keras.backend import cast
from tensorflow.contrib.keras.python.keras.backend import cast_to_floatx
from tensorflow.contrib.keras.python.keras.backend import categorical_crossentropy
from tensorflow.contrib.keras.python.keras.backend import clear_session
from tensorflow.contrib.keras.python.keras.backend import clip
from tensorflow.contrib.keras.python.keras.backend import concatenate
from tensorflow.contrib.keras.python.keras.backend import constant
from tensorflow.contrib.keras.python.keras.backend import conv1d
from tensorflow.contrib.keras.python.keras.backend import conv2d
from tensorflow.contrib.keras.python.keras.backend import conv2d_transpose
from tensorflow.contrib.keras.python.keras.backend import conv3d
from tensorflow.contrib.keras.python.keras.backend import cos
from tensorflow.contrib.keras.python.keras.backend import count_params
from tensorflow.contrib.keras.python.keras.backend import ctc_batch_cost
from tensorflow.contrib.keras.python.keras.backend import ctc_decode
from tensorflow.contrib.keras.python.keras.backend import ctc_label_dense_to_sparse
from tensorflow.contrib.keras.python.keras.backend import dot
from tensorflow.contrib.keras.python.keras.backend import dropout
from tensorflow.contrib.keras.python.keras.backend import dtype
from tensorflow.contrib.keras.python.keras.backend import elu
from tensorflow.contrib.keras.python.keras.backend import epsilon
from tensorflow.contrib.keras.python.keras.backend import equal
from tensorflow.contrib.keras.python.keras.backend import eval
from tensorflow.contrib.keras.python.keras.backend import exp
from tensorflow.contrib.keras.python.keras.backend import expand_dims
from tensorflow.contrib.keras.python.keras.backend import eye
from tensorflow.contrib.keras.python.keras.backend import flatten
from tensorflow.contrib.keras.python.keras.backend import floatx
from tensorflow.contrib.keras.python.keras.backend import foldl
from tensorflow.contrib.keras.python.keras.backend import foldr
from tensorflow.contrib.keras.python.keras.backend import function
from tensorflow.contrib.keras.python.keras.backend import gather
from tensorflow.contrib.keras.python.keras.backend import get_session
from tensorflow.contrib.keras.python.keras.backend import get_uid
from tensorflow.contrib.keras.python.keras.backend import get_value
from tensorflow.contrib.keras.python.keras.backend import gradients
from tensorflow.contrib.keras.python.keras.backend import greater
from tensorflow.contrib.keras.python.keras.backend import greater_equal
from tensorflow.contrib.keras.python.keras.backend import hard_sigmoid
from tensorflow.contrib.keras.python.keras.backend import image_data_format
from tensorflow.contrib.keras.python.keras.backend import in_test_phase
from tensorflow.contrib.keras.python.keras.backend import in_top_k
from tensorflow.contrib.keras.python.keras.backend import in_train_phase
from tensorflow.contrib.keras.python.keras.backend import int_shape
from tensorflow.contrib.keras.python.keras.backend import is_sparse
from tensorflow.contrib.keras.python.keras.backend import l2_normalize
from tensorflow.contrib.keras.python.keras.backend import learning_phase
from tensorflow.contrib.keras.python.keras.backend import less
from tensorflow.contrib.keras.python.keras.backend import less_equal
from tensorflow.contrib.keras.python.keras.backend import log
from tensorflow.contrib.keras.python.keras.backend import manual_variable_initialization
from tensorflow.contrib.keras.python.keras.backend import map_fn
from tensorflow.contrib.keras.python.keras.backend import max
from tensorflow.contrib.keras.python.keras.backend import maximum
from tensorflow.contrib.keras.python.keras.backend import mean
from tensorflow.contrib.keras.python.keras.backend import min
from tensorflow.contrib.keras.python.keras.backend import minimum
from tensorflow.contrib.keras.python.keras.backend import moving_average_update
from tensorflow.contrib.keras.python.keras.backend import name_scope
from tensorflow.contrib.keras.python.keras.backend import ndim
from tensorflow.contrib.keras.python.keras.backend import normalize_batch_in_training
from tensorflow.contrib.keras.python.keras.backend import not_equal
from tensorflow.contrib.keras.python.keras.backend import one_hot
from tensorflow.contrib.keras.python.keras.backend import ones
from tensorflow.contrib.keras.python.keras.backend import ones_like
from tensorflow.contrib.keras.python.keras.backend import permute_dimensions
from tensorflow.contrib.keras.python.keras.backend import placeholder
from tensorflow.contrib.keras.python.keras.backend import pool2d
from tensorflow.contrib.keras.python.keras.backend import pool3d
from tensorflow.contrib.keras.python.keras.backend import pow
from tensorflow.contrib.keras.python.keras.backend import print_tensor
from tensorflow.contrib.keras.python.keras.backend import prod
from tensorflow.contrib.keras.python.keras.backend import random_binomial
from tensorflow.contrib.keras.python.keras.backend import random_normal
from tensorflow.contrib.keras.python.keras.backend import random_normal_variable
from tensorflow.contrib.keras.python.keras.backend import random_uniform
from tensorflow.contrib.keras.python.keras.backend import random_uniform_variable
from tensorflow.contrib.keras.python.keras.backend import relu
from tensorflow.contrib.keras.python.keras.backend import repeat
from tensorflow.contrib.keras.python.keras.backend import repeat_elements
from tensorflow.contrib.keras.python.keras.backend import reset_uids
from tensorflow.contrib.keras.python.keras.backend import reshape
from tensorflow.contrib.keras.python.keras.backend import resize_images
from tensorflow.contrib.keras.python.keras.backend import resize_volumes
from tensorflow.contrib.keras.python.keras.backend import reverse
from tensorflow.contrib.keras.python.keras.backend import rnn
from tensorflow.contrib.keras.python.keras.backend import round
from tensorflow.contrib.keras.python.keras.backend import separable_conv2d
from tensorflow.contrib.keras.python.keras.backend import set_epsilon
from tensorflow.contrib.keras.python.keras.backend import set_floatx
from tensorflow.contrib.keras.python.keras.backend import set_image_data_format
from tensorflow.contrib.keras.python.keras.backend import set_learning_phase
from tensorflow.contrib.keras.python.keras.backend import set_session
from tensorflow.contrib.keras.python.keras.backend import set_value
from tensorflow.contrib.keras.python.keras.backend import shape
from tensorflow.contrib.keras.python.keras.backend import sigmoid
from tensorflow.contrib.keras.python.keras.backend import sign
from tensorflow.contrib.keras.python.keras.backend import sin
from tensorflow.contrib.keras.python.keras.backend import softmax
from tensorflow.contrib.keras.python.keras.backend import softplus
from tensorflow.contrib.keras.python.keras.backend import softsign
from tensorflow.contrib.keras.python.keras.backend import sparse_categorical_crossentropy
from tensorflow.contrib.keras.python.keras.backend import spatial_2d_padding
from tensorflow.contrib.keras.python.keras.backend import spatial_3d_padding
from tensorflow.contrib.keras.python.keras.backend import sqrt
from tensorflow.contrib.keras.python.keras.backend import square
from tensorflow.contrib.keras.python.keras.backend import squeeze
from tensorflow.contrib.keras.python.keras.backend import stack
from tensorflow.contrib.keras.python.keras.backend import std
from tensorflow.contrib.keras.python.keras.backend import stop_gradient
from tensorflow.contrib.keras.python.keras.backend import sum
from tensorflow.contrib.keras.python.keras.backend import switch
from tensorflow.contrib.keras.python.keras.backend import tanh
from tensorflow.contrib.keras.python.keras.backend import temporal_padding
from tensorflow.contrib.keras.python.keras.backend import to_dense
from tensorflow.contrib.keras.python.keras.backend import transpose
from tensorflow.contrib.keras.python.keras.backend import truncated_normal
from tensorflow.contrib.keras.python.keras.backend import update
from tensorflow.contrib.keras.python.keras.backend import update_add
from tensorflow.contrib.keras.python.keras.backend import update_sub
from tensorflow.contrib.keras.python.keras.backend import var
from tensorflow.contrib.keras.python.keras.backend import variable
from tensorflow.contrib.keras.python.keras.backend import zeros
from tensorflow.contrib.keras.python.keras.backend import zeros_like

del absolute_import
del division
del print_function
