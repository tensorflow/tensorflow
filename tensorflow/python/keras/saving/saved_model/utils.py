# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions shared between SavedModel saving/loading implementations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import tf_inspect


def use_wrapped_call(layer, call_fn):
  """Creates fn that adds the losses returned by call_fn & returns the outputs.

  Args:
    layer: A Keras layer object
    call_fn: tf.function that takes layer inputs (and possibly a training arg),
      and returns a tuple of (outputs, list of losses).

  Returns:
    function that calls call_fn and returns the outputs. Losses returned by
    call_fn are added to the layer losses.
  """
  training_arg_index = get_training_arg_index(layer)

  def wrapped_call(inputs, *args, **kwargs):
    """Returns the outputs from the call_fn, and adds the losses."""
    if layer._expects_training_arg:  # pylint: disable=protected-access
      training = get_training_arg(training_arg_index, args, kwargs)
      if training is None:
        training = K.learning_phase()

      args = list(args)
      kwargs = kwargs.copy()

      def replace_training_and_call(training):
        new_args, new_kwargs = set_training_arg(training, training_arg_index,
                                                args, kwargs)
        return call_fn(inputs, *new_args, **new_kwargs)

      outputs, losses = tf_utils.smart_cond(
          training,
          lambda: replace_training_and_call(True),
          lambda: replace_training_and_call(False))
    else:
      outputs, losses = call_fn(inputs)
    layer.add_loss(losses, inputs)
    return outputs
  return wrapped_call


def get_training_arg_index(layer):
  """Returns the index of 'training' in the layer call function arguments.

  Args:
    layer: Keras layer

  Returns:
    - n: index of 'training' in the call function arguments.
    - -1: if 'training' is not found in the arguments, but layer.call accepts
          variable keyword arguments
    - None: if layer doesn't expect a training argument.
  """
  if not layer._expects_training_arg:  # pylint: disable=protected-access
    return None

  arg_list = tf_inspect.getfullargspec(layer.call).args
  if tf_inspect.ismethod(layer.call):
    arg_list = arg_list[1:]
  if 'training' in arg_list:
    return arg_list.index('training')
  else:
    return -1


def set_training_arg(training, index, args, kwargs):
  if index is None:
    pass
  elif index >= 0 and len(args) > index:
    args[index] = training
  else:
    kwargs['training'] = training
  return args, kwargs


def get_training_arg(index, args, kwargs):
  if index is None:
    return None
  elif index >= 0 and len(args) > index:
    return args[index]
  else:
    return kwargs.get('training', None)
