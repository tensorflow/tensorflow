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
"""Methods related to optimizers used in canned_estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


from tensorflow.python.training import adagrad
from tensorflow.python.training import adam
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import rmsprop


_OPTIMIZER_CLS_NAMES = {
    'Adagrad': adagrad.AdagradOptimizer,
    'Adam': adam.AdamOptimizer,
    'Ftrl': ftrl.FtrlOptimizer,
    'RMSProp': rmsprop.RMSPropOptimizer,
    'SGD': gradient_descent.GradientDescentOptimizer,
}


def get_optimizer_instance(opt, learning_rate=None):
  """Returns an optimizer instance.

  Supports the following types for the given `opt`:
  * An `Optimizer` instance: Returns the given `opt`.
  * A string: Creates an `Optimizer` subclass with the given `learning_rate`.
    Supported strings:
    * 'Adagrad': Returns an `AdagradOptimizer`.
    * 'Adam': Returns an `AdamOptimizer`.
    * 'Ftrl': Returns an `FtrlOptimizer`.
    * 'RMSProp': Returns an `RMSPropOptimizer`.
    * 'SGD': Returns a `GradientDescentOptimizer`.

  Args:
    opt: An `Optimizer` instance, or string, as discussed above.
    learning_rate: A float. Only used if `opt` is a string.

  Returns:
    An `Optimizer` instance.

  Raises:
    ValueError: If `opt` is an unsupported string.
    ValueError: If `opt` is a supported string but `learning_rate` was not
      specified.
    ValueError: If `opt` is none of the above types.
  """
  if isinstance(opt, six.string_types):
    if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
      if not learning_rate:
        raise ValueError('learning_rate must be specified when opt is string.')
      return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
    raise ValueError(
        'Unsupported optimizer name: {}. Supported names are: {}'.format(
            opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))
  if callable(opt):
    opt = opt()
  if not isinstance(opt, optimizer_lib.Optimizer):
    raise ValueError(
        'The given object is not an Optimizer instance. Given: {}'.format(opt))
  return opt
