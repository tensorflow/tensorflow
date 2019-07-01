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
  # TODO(kathywu): Support mask argument and multi-input call functions.
  def wrapped_call(inputs, **kwargs):
    """Returns the outputs from the call_fn, and adds the losses."""
    if layer._expects_training_arg:  # pylint: disable=protected-access
      training = kwargs.pop('training', None)
      if training is None:
        training = K.learning_phase()
      outputs, losses = tf_utils.smart_cond(
          training,
          lambda: call_fn(inputs, training=True),
          lambda: call_fn(inputs, training=False))
    else:
      outputs, losses = call_fn(inputs)
    layer.add_loss(losses, inputs)
    return outputs
  return wrapped_call
