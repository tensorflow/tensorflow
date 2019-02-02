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
"""TfLite BasicRnnCell wrapper.

TODO(renjieliu): Find a better home for this one.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools

from tensorflow.lite.python import lite
from tensorflow.python.keras import activations
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl


class TfLiteRNNCell(rnn_cell_impl.LayerRNNCell):
  """The most basic RNN cell.

  This is used only for TfLite, it provides hints and it also makes the
  variables in the desired for the tflite ops.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    """Initializes the parameters for an RNN cell.

    Args:
      num_units: int, The number of units in the RNN cell.
      activation: Nonlinearity to use.  Default: `tanh`. It could also be string
        that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope. Raises an error if not `True` and the existing scope
        already has the given variables.
      name: String, the name of the layer. Layers with the same name will share
        weights, but to avoid mistakes we require reuse=True in such cases.
      dtype: Default dtype of the layer (default of `None` means use the type of
        the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().

    Raises:
      ValueError: If the existing scope already has the given variables.
    """
    super(TfLiteRNNCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)

    # Inputs must be Rank-2.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._tflite_wrapper = lite.OpHint("UnidirectionalSequenceRnn")
    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    """Builds the RNN cell.

    Args:
      inputs_shape: Rnn input tensor shape.

    Raises:
      ValueError: If last dimension of the input shape is not known.
    """
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       (inputs_shape,))

    input_depth = inputs_shape[-1]

    def add_variable_wrapped(name, shape, initializer, index):
      var = self.add_variable(name, shape=shape, initializer=initializer)
      return self._tflite_wrapper.add_input(
          var, name=name, index_override=index)

    self._input_weights = add_variable_wrapped(
        "input_weights", [self._num_units, input_depth], None, 1)
    self._recurrent_weights = add_variable_wrapped(
        "recurrent_weights", [self._num_units, self._num_units], None, 2)
    self._bias = add_variable_wrapped(
        "bias",
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype),
        index=3)

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    inputs = self._tflite_wrapper.add_input(
        inputs, tag="input", name="input", aggregate="stack", index_override=0)
    state = self._tflite_wrapper.add_input(
        state,
        tag="hidden_state",
        name="hidden_state",
        aggregate="first",
        index_override=4)
    weights = array_ops.transpose(
        array_ops.concat([self._input_weights, self._recurrent_weights], 1))
    gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), weights)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    output = self._tflite_wrapper.add_output(
        output,
        tag="output",
        name="output",
        index_override=1,
        aggregate="stack")
    return output, output

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(TfLiteRNNCell, self).get_config()
    return dict(itertools.chain(base_config.items(), config.items()))
