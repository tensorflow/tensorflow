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
"""Mock DataFrame constituents for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta

import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.python.framework import tensor_shape

# TODO(soergel): Consider cleaning this up using tf.test.mock


class MockTensor(object):
  """A mock Tensor for use in testing."""

  def __init__(self, name, dtype):
    super(MockTensor, self).__init__()
    self._name = name
    self._dtype = dtype

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  def get_shape(self):
    return tensor_shape.unknown_shape()

  def __repr__(self):
    return "Mock Tensor: %s" % self._name

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)


class MockSparseTensor(object):
  """A mock SparseTensor for use in testing."""

  def __init__(self, name, dtype):
    super(MockSparseTensor, self).__init__()
    self._name = name
    self._dtype = dtype
    self._shape = tensor_shape.unknown_shape()
    self.indices = MockTensor("%s indices" % name, tf.int32)
    self.values = MockTensor("%s values" % name, dtype)

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  def get_shape(self):
    return self._shape

  def __repr__(self):
    return "Mock SparseTensor: %s" % self._name

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)


class MockSeries(learn.Series):
  """A mock series for use in testing."""

  def __init__(self, cachekey, mock_tensor):
    super(MockSeries, self).__init__()
    self._cachekey = cachekey
    self._mock_tensor = mock_tensor

  def build(self, cache):
    return self._mock_tensor

  def __repr__(self):
    return self._cachekey


class MockTransform(learn.Transform):
  """A mock transform for use in testing."""

  __metaclass__ = ABCMeta

  def __init__(self, param_one, param_two):
    super(MockTransform, self).__init__()
    self._param_one = param_one
    self._param_two = param_two

  @property
  def name(self):
    return "MockTransform"

  @learn.parameter
  def param_one(self):
    return self._param_one

  @learn.parameter
  def param_two(self):
    return self._param_two

  @property
  def input_valency(self):
    return 1

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)


class MockZeroOutputTransform(MockTransform):
  """A mock transform for use in testing."""

  _mock_output_names = []

  def __init__(self, param_one, param_two):
    super(MockZeroOutputTransform, self).__init__(param_one, param_two)

  @property
  def _output_names(self):
    return MockZeroOutputTransform._mock_output_names

  def _apply_transform(self, input_tensors):
    # pylint: disable=not-callable
    return self.return_type()


class MockOneOutputTransform(MockTransform):
  """A mock transform for use in testing."""

  _mock_output_names = ["out1"]

  def __init__(self, param_one, param_two):
    super(MockOneOutputTransform, self).__init__(param_one, param_two)

  @property
  def _output_names(self):
    return MockOneOutputTransform._mock_output_names

  def _apply_transform(self, input_tensors):
    # pylint: disable=not-callable
    return self.return_type(MockTensor("Mock Tensor 1", tf.int32))


class MockTwoOutputTransform(MockTransform):
  """A mock transform for use in testing."""

  _mock_output_names = ["out1", "out2"]

  @learn.parameter
  def param_three(self):
    return self._param_three

  def __init__(self, param_one, param_two, param_three):
    super(MockTwoOutputTransform, self).__init__(param_one, param_two)
    self._param_three = param_three

  @property
  def _output_names(self):
    return MockTwoOutputTransform._mock_output_names

  def _apply_transform(self, input_tensors):
    # pylint: disable=not-callable
    return self.return_type(
        MockTensor("Mock Tensor 1", tf.int32),
        MockTensor("Mock Tensor 2", tf.int32))


class Mock2x2Transform(MockTransform):
  """A mock transform for use in testing."""

  _mock_output_names = ["out1", "out2"]

  def __init__(self, param_one, param_two, param_three):
    super(Mock2x2Transform, self).__init__(param_one, param_two)
    self._param_three = param_three

  @property
  def _output_names(self):
    return Mock2x2Transform._mock_output_names

  def _apply_transform(self, input_tensors):
    # pylint: disable=not-callable
    return self.return_type(
        MockTensor("Out " + self._param_one, tf.int32),
        MockTensor("Out " + self._param_two, tf.int32))

  @property
  def input_valency(self):
    return 2
