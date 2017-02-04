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

"""Utilities for creating input_fns."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops


# A return type allowing input_fns to return multiple values in a well-
# defined way (analogous to ModelFnOps).
# The expected return values are:
#  features: a dict of string to `Tensor` or `SparseTensor`, giving the features
#            to be passed to the model.
#  labels: a dict of string to `Tensor` or `SparseTensor`, giving labels (aka
#            targets) for training.
#  default_inputs: a dict of string to `Tensor` or `SparseTensor`, giving the
#            input placeholders (if any) that this input_fn expects to be fed.
InputFnOps = collections.namedtuple('InputFnOps',
                                    ['features',
                                     'labels',
                                     'default_inputs'])


def build_parsing_serving_input_fn(feature_spec, default_batch_size=None):
  """Build an input_fn appropriate for serving, expecting fed tf.Examples.

  Creates an input_fn that expects a serialized tf.Example fed into a string
  placeholder.  The function parses the tf.Example according to the provided
  feature_spec, and returns all parsed Tensors as features.  This input_fn is
  for use at serving time, so the labels return value is always None.

  Args:
    feature_spec: a dict of string to `VarLenFeature`/`FixedLenFeature`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    An input_fn suitable for use in serving.
  """
  def input_fn():
    """An input_fn that expects a serialized tf.Example."""
    serialized_tf_example = array_ops.placeholder(dtype=dtypes.string,
                                                  shape=[default_batch_size],
                                                  name='input_example_tensor')
    inputs = {'examples': serialized_tf_example}
    features = parsing_ops.parse_example(serialized_tf_example, feature_spec)
    labels = None  # these are not known in serving!
    return InputFnOps(features, labels, inputs)
  return input_fn


def build_default_serving_input_fn(features, default_batch_size=None):
  """Build an input_fn appropriate for serving, expecting feature Tensors.

  Creates an input_fn that expects all features to be fed directly.
  This input_fn is for use at serving time, so the labels return value is always
  None.

  Args:
    features: a dict of string to `Tensor`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    An input_fn suitable for use in serving.
  """
  def input_fn():
    """an input_fn that expects all features to be fed directly."""
    features_placeholders = {}
    for name, t in features.items():
      shape_list = t.get_shape().as_list()
      shape_list[0] = default_batch_size
      shape = tensor_shape.TensorShape(shape_list)

      features_placeholders[name] = array_ops.placeholder(dtype=t.dtype,
                                                          shape=shape,
                                                          name=t.name)
    labels = None  # these are not known in serving!
    return InputFnOps(features_placeholders, labels, features_placeholders)
  return input_fn
