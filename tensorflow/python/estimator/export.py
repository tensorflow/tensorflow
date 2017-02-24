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
"""Configuration describing how inputs will be received at serving time."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor


SINGLE_FEATURE_DEFAULT_NAME = 'feature'
SINGLE_RECEIVER_DEFAULT_NAME = 'input'


class ServingInputReceiver(collections.namedtuple('ServingInputReceiver',
                                                  ['features',
                                                   'receiver_tensors'])):
  """A return type for a serving_input_receiver_fn.

  The expected return values are:
    features: A dict of string to `Tensor` or `SparseTensor`, specifying the
      features to be passed to the model.
    receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
      input nodes where this receiver expects to be fed.  Typically, this is a
      single placeholder expecting serialized `tf.Example` protos.
  """
  # TODO(soergel): add receiver_alternatives when supported in serving.

  def __new__(cls, features, receiver_tensors):
    if features is None:
      raise ValueError('features must be defined.')
    if not isinstance(features, dict):
      features = {SINGLE_FEATURE_DEFAULT_NAME: features}
    for name, tensor in features.items():
      if not isinstance(name, str):
        raise ValueError('feature keys must be strings: {}.'.format(name))
      if not (isinstance(tensor, ops.Tensor)
              or isinstance(tensor, sparse_tensor.SparseTensor)):
        raise ValueError(
            'feature {} must be a Tensor or SparseTensor.'.format(name))

    if receiver_tensors is None:
      raise ValueError('receiver_tensors must be defined.')
    if not isinstance(receiver_tensors, dict):
      receiver_tensors = {SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors}
    for name, tensor in receiver_tensors.items():
      if not isinstance(name, str):
        raise ValueError(
            'receiver_tensors keys must be strings: {}.'.format(name))
      if not isinstance(tensor, ops.Tensor):
        raise ValueError(
            'receiver_tensor {} must be a Tensor.'.format(name))

    return super(ServingInputReceiver, cls).__new__(
        cls, features=features, receiver_tensors=receiver_tensors)
