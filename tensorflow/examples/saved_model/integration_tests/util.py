# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf

from tensorflow.python.framework import smart_cond
from tensorflow.python.util import tf_inspect


# TODO(vbardiovsky): We should just reuse Keras's Lambda layer, when that
# enables to get trainable variables.
class CustomLayer(tf.keras.layers.Layer):
  """Wraps callable object as a `Layer` object.

  Args:
    func: The callable object to wrap. Layer inputs are passed as the first
      positional argument. If `func` accepts a `training` argument, a Python
      boolean is passed for it.
      If present, the following attributes of `func` have a special meaning:
        * variables: a list of all tf.Variable objects that `func` depends on.
        * trainable_variables: those elements of `variables` that are reported
          as trainable variables of this Keras Layer.
        * regularization_losses: a list of callables to be added as losses
          of this Keras layer. Each one must accept zero arguments and return
          a scalare tensor.
    trainable: Boolean controlling whether the trainable variables of `func`
      are reported as trainable variables of this layer.
    arguments: optionally, a dict with additional keyword arguments passed
      to `func`.
    **kwargs: 'output_shape': A tuple with the (possibly partial) output
      shape of the callable *without* leading batch size. Other arguments
      are pass into the Layer constructor.
  """

  def __init__(self, func, trainable=False, arguments=None, **kwargs):
    # Set self._{non,}_trainable_weights before calling Layer.__init__.
    if hasattr(func, 'trainable_variables'):
      self._trainable_weights = [v for v in func.trainable_variables]
      trainable_variables_set = set(func.trainable_variables)
    else:
      self._trainable_weights = []
      trainable_variables_set = set()
    if hasattr(func, 'variables'):
      self._non_trainable_weights = [v for v in func.variables
                                     if v not in trainable_variables_set]
    else:
      self._non_trainable_weights = []  # TODO(arnoegw): Infer from `func`.

    # TODO(b/124219898): We should be able to get the embedding dimension from
    # the restored model.
    if 'output_shape' in kwargs:
      self._output_shape = tuple(kwargs.pop('output_shape'))

    super(CustomLayer, self).__init__(trainable=trainable, **kwargs)
    # Prepare to call `func`.
    self._func = func
    self._func_fullargspec = tf_inspect.getfullargspec(func.__call__)
    self._func_wants_training = (
        'training' in self._func_fullargspec.args or
        'training' in self._func_fullargspec.kwonlyargs)
    self._arguments = arguments or {}
    # Forward the callable's regularization losses (if any).
    if hasattr(func, 'regularization_losses'):
      for l in func.regularization_losses:
        if not callable(l):
          raise ValueError(
              'CustomLayer(func) expects func.regularization_losses to be an '
              'iterable of callables, each returning a scalar loss term.')
        self.add_loss(l)  # Supports callables.

  def call(self, x, training=None):
    # We basically want to call this...
    f = functools.partial(self._func, x, **self._arguments)
    # ...but we may also have to pass a Python boolean for `training`.
    if not self._func_wants_training:
      result = f()
    else:
      if training is None:
        training = tf.keras.backend.learning_phase()  # Could be a tensor.
      result = smart_cond.smart_cond(training,
                                     lambda: f(training=True),
                                     lambda: f(training=False))
    # TODO(b/124219898): Polymorphic function should return shaped tensor.
    if hasattr(self, '_output_shape'):
      result.set_shape((x.shape[0],) + self._output_shape)
    return result
