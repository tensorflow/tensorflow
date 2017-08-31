# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental API for TensorFlow's "Eager" mode of execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# TODO(agarwal): get rid of this import and change callers to use the classes in
# ops.py.
# pylint: disable=unused-import
from tensorflow.python.framework.ops import _tensor_from_handle
from tensorflow.python.framework.ops import EagerTensor as Tensor
# pylint: enable=unused-import


class _Op(object):
  """Fake op for _LazyZero to make its python API tf.Tensor-like."""

  def __init__(self):
    self.type = "Zeros"


class LazyZero(object):
  """Lazily-instantiated zero-valued Tensor used as autograd accumulator."""

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype
    self.op = _Op()

  def __add__(self, other):
    return other

  def __radd__(self, other):
    return other

  def numpy(self):
    return np.zeros(self.shape, self.dtype)
