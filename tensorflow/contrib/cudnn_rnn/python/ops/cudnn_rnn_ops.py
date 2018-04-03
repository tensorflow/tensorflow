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
"""Cudnn RNN operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import saver

CUDNN_RNN_UNIDIRECTION = "unidirectional"
CUDNN_RNN_BIDIRECTION = "bidirectional"
CUDNN_LSTM = "lstm"
CUDNN_GRU = "gru"
CUDNN_RNN_RELU = "rnn_relu"
CUDNN_RNN_TANH = "rnn_tanh"

# Half for cell input, half for hidden states.
CUDNN_LSTM_PARAMS_PER_LAYER = 8
CUDNN_GRU_PARAMS_PER_LAYER = 6
CUDNN_RNN_TANH_PARAMS_PER_LAYER = 2
CUDNN_RNN_RELU_PARAMS_PER_LAYER = 2

CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_INPUT_SKIP_MODE = "skip_input"
CUDNN_INPUT_AUTO_MODE = "auto_select"

# pylint:disable=protected-access
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME
# pylint:enable=protected-access


class CudnnCompatibleLSTMCell(lstm_ops.LSTMBlockCell):
  """Cudnn Compatible LSTMCell.

  A simple wrapper around @{tf.contrib.rnn.LSTMBlockCell} to use along with
  @{tf.contrib.cudnn_rnn.CudnnLSTM}. The latter's params can be used by
  this cell seamlessly.
  """

  def __init__(self, num_units, reuse=None):
    super(CudnnCompatibleLSTMCell, self).__init__(
        num_units, forget_bias=0, cell_clip=None, use_peephole=False,
        reuse=reuse, name="cudnn_compatible_lstm_cell")
    self._names.update({"scope": "cudnn_compatible_lstm_cell"})


class CudnnCompatibleGRUCell(rnn_cell_impl.GRUCell):
  """Cudnn Compatible GRUCell.

  A GRU impl akin to @{tf.nn.rnn_cell.GRUCell} to use along with
  @{tf.contrib.cudnn_rnn.CudnnGRU}. The latter's params can be used by
  it seamlessly.

  It differs from platform-independent GRUs in how the new memory gate is
  calculated. Nvidia picks this variant based on GRU author's[1] suggestion and
  the fact it has no accuracy impact[2].
  [1] https://arxiv.org/abs/1406.1078
  [2] http://svail.github.io/diff_graphs/

  Cudnn compatible GRU (from Cudnn library user guide):
  ```python
  # reset gate
  $$r_t = \sigma(x_t * W_r + h_t-1 * R_h + b_{Wr} + b_{Rr})$$
  # update gate
  $$u_t = \sigma(x_t * W_u + h_t-1 * R_u + b_{Wu} + b_{Ru})$$
  # new memory gate
  $$h'_t = tanh(x_t * W_h + r_t .* (h_t-1 * R_h + b_{Rh}) + b_{Wh})$$
  $$h_t = (1 - u_t) .* h'_t + u_t .* h_t-1$$
  ```

  Other GRU (see @{tf.nn.rnn_cell.GRUCell} and @{tf.contrib.rnn.GRUBlockCell}):
  ```python
  # new memory gate
  \\(h'_t = tanh(x_t * W_h + (r_t .* h_t-1) * R_h + b_{Wh})\\)
  ```
  which is not equivalent to Cudnn GRU: in addition to the extra bias term b_Rh,
  ```python
  \\(r .* (h * R) != (r .* h) * R\\)
  ```
  """

  def __init__(self, num_units, reuse=None, kernel_initializer=None):
    super(CudnnCompatibleGRUCell, self).__init__(
        num_units,
        activation=None,
        reuse=reuse,
        kernel_initializer=kernel_initializer)

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))

    self._candidate_input_kernel = self.add_variable(
        "candidate/input_projection/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_hidden_kernel = self.add_variable(
        "candidate/hidden_projection/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_units],
        initializer=self._kernel_initializer)

    self._candidate_input_bias = self.add_variable(
        "candidate/input_projection/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))
    self._candidate_hidden_bias = self.add_variable(
        "candidate/hidden_projection/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    candidate = nn_ops.bias_add(
        math_ops.matmul(inputs, self._candidate_input_kernel),
        self._candidate_input_bias)
    candidate += r * nn_ops.bias_add(
        math_ops.matmul(state, self._candidate_hidden_kernel),
        self._candidate_hidden_bias)
    candidate = self._activation(candidate)
    new_h = (1-u) * candidate + u * state
    return new_h, new_h


# TODO(yaozhang): make sure we only save the canonical version of params and
# don't save the platform-specific version to avoid potential race
# conditions where params is updated by both versions when being restored.
# Currently, checkpointing will function properly, despite that we save both
# versions, because Saver restores customized savables after Variables.
# However, it is good to not rely on this restoring order of Saver and to
# avoid unnecessary storage. Add a test to check only the canonical version is
# saved.
class CudnnOpaqueParamsSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Abstract SaveableObject implementation handling Cudnn opaque params."""

  def __init__(self,
               opaque_params,
               num_layers,
               num_units,
               input_size,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               scope=None,
               name="cudnn_rnn_saveable"):
    """Creates a CudnnOpaqueParamsSaveable object.

       CudnnOpaqueParamsSaveable is saveable/restorable in a checkpoint file
       and is used to save/restore the weights and biases parameters in a
       canonical format which is directly consumable by platform-independent tf
       RNN cells. Parameters are saved as tensors layer by layer with weight
       tensors followed by bias tensors, and forward direction followed by
       backward direction (if applicable). When restoring, a user could name
       param_variables as desired, and restore weight and bias tensors to these
       variables.

       For CudnnRNNRelu or CudnnRNNTanh, there are 2 tensors per weight and per
       bias for each layer: tensor 0 is applied to the input from the previous
       layer and tensor 1 to the recurrent input.

       For CudnnLSTM, there are 8 tensors per weight and per bias for each
       layer: tensor 0-3 are applied to the input from the previous layer and
       tensor 4-7 to the recurrent input. Tensor 0 and 4 are for the input gate;
       tensor 1 and 5 the forget gate; tensor 2 and 6 the new memory gate;
       tensor 3 and 7 the output gate.

       For CudnnGRU, there are 6 tensors per weight and per bias for each layer:
       tensor 0-2 are applied to the input from the previous layer and
       tensor 3-5 to the recurrent input. Tensor 0 and 3 are for the reset gate;
       tensor 1 and 4 the update gate; tensor 2 and 5 the new memory gate.

    Args:
      opaque_params: a variable, Cudnn RNN opaque params.
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and the actual computation before the first layer. It could be
          'linear_input', 'skip_input' or 'auto_select'.
          'linear_input' (default) always applies a linear projection of input
          onto RNN hidden state. (standard RNN behavior).
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      scope: string of VariableScope, the scope of equivalent subgraph
          consisting only platform-independent tf RNN cells.
      name: the name of the CudnnOpaqueParamsSaveable object.
    """
    # Define in subclasses.
    self._num_layers = num_layers
    self._input_size = input_size
    self._num_units = num_units
    self._input_mode = input_mode
    self._direction = direction
    if scope is not None:
      scope_name = scope.name if isinstance(scope, vs.VariableScope) else scope
      self._scope = scope_name or None
    else:
      self._scope = None

    self._variables = opaque_params
    self._num_dirs = 1 if self._direction == CUDNN_RNN_UNIDIRECTION else 2
    self._num_params = (
        self._num_params_per_layer * self._num_layers * self._num_dirs)

    weights, biases = self._OpaqueParamsToCanonical()
    (weights, weight_names), (biases, bias_names) = self._TransformCanonical(
        weights, biases)
    # We currently don't use slice_spec. It might be useful in a distributed
    # setting where each parameter server node stores a slice of variable,
    # instead of having the master pull all slices and then save them.
    slice_spec = ""
    params = weights + biases
    param_names = weight_names + bias_names
    if self._scope:
      param_names = ["%s/%s" % (self._scope, pn) for pn in param_names]

    specs = [
        saver.BaseSaverBuilder.SaveSpec(param, slice_spec, param_name)
        for param, param_name in zip(params, param_names)
    ]
    super(CudnnOpaqueParamsSaveable, self).__init__(
        array_ops.identity(self._variables), specs, name)

  def restore(self, restored_tensors, restored_shapes):
    weights, biases = self._ReverseTransformCanonical(restored_tensors)
    weights = [array_ops.reshape(w, [-1]) for w in weights]
    opaque_params = self._CanonicalToOpaqueParams(weights, biases)

    return state_ops.assign(
        self._variables, opaque_params, validate_shape=False)

  def _TFCanonicalNamePrefix(self, layer, is_fwd=True):
    if self._direction == CUDNN_RNN_UNIDIRECTION:
      return "rnn/multi_rnn_cell/cell_%d/%s" % (layer, self._rnn_cell_name)
    else:
      if is_fwd:
        return ("stack_bidirectional_rnn/cell_%d/bidirectional_rnn/fw/%s" %
                (layer, self._rnn_cell_name))
      else:
        return ("stack_bidirectional_rnn/cell_%d/bidirectional_rnn/bw/%s" %
                (layer, self._rnn_cell_name))

  def _OpaqueParamsToCanonical(self):
    """Converts opaque params to Cudnn canonical format.

    Returns:
      2 list for weights and biases respectively.
    """
    with ops.device("/gpu:0"):
      weights, biases = gen_cudnn_rnn_ops.cudnn_rnn_params_to_canonical(
          num_layers=self._num_layers,
          num_units=self._num_units,
          input_size=self._input_size,
          params=self._variables,
          num_params=self._num_params,
          rnn_mode=self._rnn_mode,
          input_mode=self._input_mode,
          direction=self._direction)
      return (weights, biases)

  def _CanonicalToOpaqueParams(self, cu_weights, cu_biases):
    """Converts from Cudnn canonical format to opaque params.

    Args:
      cu_weights: a list of tensors, Cudnn canonical weights.
      cu_biases: a list of tensors, Cudnn canonical biases.
    Returns:
      a single opaque tensor.
    """
    with ops.device("/gpu:0"):
      return gen_cudnn_rnn_ops.cudnn_rnn_canonical_to_params(
          num_layers=self._num_layers,
          num_units=self._num_units,
          input_size=self._input_size,
          weights=cu_weights,
          biases=cu_biases,
          rnn_mode=self._rnn_mode,
          input_mode=self._input_mode,
          direction=self._direction)

  def _TransformCanonical(self, cu_weights, cu_biases):
    r"""Transform from Cudnn canonical to tf canonical.

    The elements of argument lists are laid out in the following format:
        ------------------------------------------------------------
        | weights                    | biases                      |
        ------------------------------------------------------------
        \                             \
         \                             \
          -------------------------------
          | layer1     |layer2     |... |
          -------------------------------
          \             \
           ---------------
           |fwd   |bak   |
           ---------------
    Args:
      cu_weights: a list of tensors of Cudnn canonical weights.
      cu_biases: a list of tensors of Cudnn canonical biases.
    Returns:
      2 tuples, one for weights and the other for bias.
      Each tuple has two lists: the 1st for transformed tf canonical tensors
      and the 2nd for the names of the tensors under which they are saved.
    """
    tf_weights, tf_biases = [], []
    tf_weights_names, tf_bias_names = [], []

    layer_weights_num = self._num_params_per_layer * self._num_dirs
    layer_biases_num = layer_weights_num

    for i in range(self._num_layers):
      layer_weights = cu_weights[i * layer_weights_num:
                                 (i + 1) * layer_weights_num]
      layer_biases = cu_biases[i * layer_biases_num:(i + 1) * layer_biases_num]
      if self._direction == CUDNN_RNN_UNIDIRECTION:
        prefix = self._TFCanonicalNamePrefix(i)
        self._TransformSingleLayerCanonical(layer_weights, layer_biases, prefix,
                                            tf_weights, tf_weights_names,
                                            tf_biases, tf_bias_names)
      else:
        fw_prefix = self._TFCanonicalNamePrefix(i, is_fwd=True)
        bw_prefix = self._TFCanonicalNamePrefix(i, is_fwd=False)

        fw_weights = layer_weights[:len(layer_weights) // 2]
        bw_weights = layer_weights[len(layer_weights) // 2:]
        fw_biases = layer_biases[:len(layer_biases) // 2]
        bw_biases = layer_biases[len(layer_biases) // 2:]

        self._TransformSingleLayerCanonical(fw_weights, fw_biases, fw_prefix,
                                            tf_weights, tf_weights_names,
                                            tf_biases, tf_bias_names)

        self._TransformSingleLayerCanonical(bw_weights, bw_biases, bw_prefix,
                                            tf_weights, tf_weights_names,
                                            tf_biases, tf_bias_names)
    return (tf_weights, tf_weights_names), (tf_biases, tf_bias_names)

  def _TransformSingleLayerCanonical(self, cu_weights, cu_biases, prefix,
                                     tf_weights, tf_weights_names, tf_biases,
                                     tf_bias_names):
    r"""Transform single layer Cudnn canonicals to tf canonicals.

    The elements of cu_weights, cu_biases are laid out in the following format:
    -------------------------------------------------------------------------
    | gate0 param on inputs | gate0 param on hidden state | gate1 ..........|
    -------------------------------------------------------------------------
    Args:
      cu_weights: a list of tensors, single layer weights.
      cu_biases: a list of tensors, single layer biases.
      prefix: the shared prefix of all tensor names.
      tf_weights: a list where transformed weights are stored.
      tf_weights_names: a list where names of transformed weights are stored.
      tf_biases: a list where transformed biases are stored.
      tf_bias_names: a list where names of transformed biases are stored.
    """
    raise NotImplementedError("Abstract method")

  def _ReverseTransformCanonical(self, tf_canonicals):
    r"""Transform from tf canonical to Cudnn canonical.

    This is the reverse routine of _TransformCanonical().
    Args:
      tf_canonicals: a list of tensors of tf canonical params. The elements are
        laid out in the following format:
        ------------------------------------------------------------
        | weights                    | biases                      |
        ------------------------------------------------------------
        \                             \
         \                             \
          -------------------------------
          | layer1     |layer2     |... |
          -------------------------------
          \             \
           ---------------
           |fwd   |bak   |
           ---------------
    Returns:
      2 lists: the recovered cudnn canonical weights and biases.
    """
    weights = tf_canonicals[:len(tf_canonicals) // 2]
    biases = tf_canonicals[len(tf_canonicals) // 2:]

    cu_weights, cu_biases = [], []
    layer_weights_num = len(weights) // self._num_layers
    layer_biases_num = len(biases) // self._num_layers
    for i in range(self._num_layers):
      layer_weights = weights[i * layer_weights_num:(i + 1) * layer_weights_num]
      layer_biases = biases[i * layer_biases_num:(i + 1) * layer_biases_num]
      if self._direction == CUDNN_RNN_UNIDIRECTION:
        cu_weights.extend(self._tf_to_cudnn_weights(i, *layer_weights))
        cu_biases.extend(self._tf_to_cudnn_biases(*layer_biases))
      else:
        fw_weights, bw_weights = layer_weights[:len(
            layer_weights) // 2], layer_weights[len(layer_weights) // 2:]
        fw_biases, bw_biases = layer_biases[:len(
            layer_biases) // 2], layer_biases[len(layer_biases) // 2:]
        cu_weights.extend(self._tf_to_cudnn_weights(i, *fw_weights))
        cu_biases.extend(self._tf_to_cudnn_biases(*fw_biases))

        cu_weights.extend(self._tf_to_cudnn_weights(i, *bw_weights))
        cu_biases.extend(self._tf_to_cudnn_biases(*bw_biases))
    return cu_weights, cu_biases

  def _cudnn_to_tf_weights(self, *cu_weights):
    r"""Stitching cudnn canonical weights to generate tf canonical weights."""
    raise NotImplementedError("Abstract method")

  def _tf_to_cudnn_weights(self, layer, *tf_weights):
    r"""Reverse the operations in StitchWeights()."""
    raise NotImplementedError("Abstract method")

  def _cudnn_to_tf_biases(self, *biases):
    r"""Stitching cudnn canonical biases to generate tf canonical biases."""
    raise NotImplementedError("Abstract method")

  def _tf_to_cudnn_biases(self, *tf_biases):
    r"""Reverse the operations in StitchBiases()."""
    raise NotImplementedError("Abstract method")


class CudnnLSTMSaveable(CudnnOpaqueParamsSaveable):
  """SaveableObject implementation handling Cudnn LSTM opaque params."""

  _rnn_mode = CUDNN_LSTM
  _num_params_per_layer = CUDNN_LSTM_PARAMS_PER_LAYER

  # pylint:disable=protected-access
  _rnn_cell_name = base_layer._to_snake_case(CudnnCompatibleLSTMCell.__name__)

  # pylint:enable=protected-access

  def _cudnn_to_tf_gate_params(self, *cu_gate_order):
    i_g, f_g, c_g, o_g = cu_gate_order
    return [i_g, c_g, f_g, o_g]

  def _tf_to_cudnn_gate_params(self, *tf_gate_order):
    i_g, c_g, f_g, o_g = tf_gate_order
    return [i_g, f_g, c_g, o_g]

  def _cudnn_to_tf_weights(self, *cu_weights):
    r"""Stitching cudnn canonical weights to generate tf canonical weights."""
    w_i, w_f, w_c, w_o, r_i, r_f, r_c, r_o = cu_weights

    # pylint: disable=invalid-name
    W_i = array_ops.concat([w_i, r_i], axis=1)
    W_f = array_ops.concat([w_f, r_f], axis=1)
    W_c = array_ops.concat([w_c, r_c], axis=1)
    W_o = array_ops.concat([w_o, r_o], axis=1)
    # pylint: enable=invalid-name
    # Cudnn LSTM weights are in ifco order, other tf LSTMs are in icfo order.
    reordered = self._cudnn_to_tf_gate_params(* [W_i, W_f, W_c, W_o])
    return (array_ops.transpose(array_ops.concat(reordered, axis=0)),)

  def _tf_to_cudnn_weights(self, layer, *tf_weights):
    r"""Reverse the operations in StitchWeights()."""
    input_size = self._input_size
    num_units = self._num_units
    if layer == 0:
      input_weight_width = input_size
    else:
      input_weight_width = num_units
      if self._direction == CUDNN_RNN_BIDIRECTION:
        input_weight_width *= 2

    (tf_weight,) = tf_weights
    w = array_ops.transpose(tf_weight)
    # pylint: disable=invalid-name
    W_i, W_f, W_c, W_o = self._tf_to_cudnn_gate_params(*array_ops.split(
        w, 4, axis=0))

    w_i, r_i = array_ops.split(W_i, [input_weight_width, num_units], axis=1)
    w_c, r_c = array_ops.split(W_c, [input_weight_width, num_units], axis=1)
    w_f, r_f = array_ops.split(W_f, [input_weight_width, num_units], axis=1)
    w_o, r_o = array_ops.split(W_o, [input_weight_width, num_units], axis=1)
    return w_i, w_f, w_c, w_o, r_i, r_f, r_c, r_o
    # pylint: enable=invalid-name

  def _cudnn_to_tf_biases(self, *cu_biases):
    r"""Stitching cudnn canonical biases to generate tf canonical biases."""
    b_wi, b_wf, b_wc, b_wo, b_ri, b_rf, b_rc, b_ro = cu_biases
    # Save only the sum instead of individual biases. When recovering, return
    # two biases each with half the value. Since RNN does not regularize by
    # weight decay, it has no side effect in training or inference.
    # pylint: disable=invalid-name
    B_i = b_wi + b_ri
    B_f = b_wf + b_rf
    B_c = b_wc + b_rc
    B_o = b_wo + b_ro
    # pylint: enable=invalid-name
    reordered = self._cudnn_to_tf_gate_params(* [B_i, B_f, B_c, B_o])
    return (array_ops.concat(reordered, axis=0),)

  def _tf_to_cudnn_biases(self, *tf_biases):
    r"""Reverse the operations in StitchBiases()."""
    (tf_bias,) = tf_biases
    # pylint: disable=invalid-name
    B_i, B_f, B_c, B_o = self._tf_to_cudnn_gate_params(*array_ops.split(
        tf_bias, 4, axis=0))
    # pylint: enable=invalid-name
    # pylint: disable=unbalanced-tuple-unpacking
    b_wi, b_ri = (B_i * 0.5,) * 2
    b_wf, b_rf = (B_f * 0.5,) * 2
    b_wc, b_rc = (B_c * 0.5,) * 2
    b_wo, b_ro = (B_o * 0.5,) * 2
    # pylint: enable=unbalanced-tuple-unpacking
    # Return ifco order for Cudnn LSTM.
    return b_wi, b_wf, b_wc, b_wo, b_ri, b_rf, b_rc, b_ro

  def _TransformSingleLayerCanonical(self, weights, biases, prefix, tf_weights,
                                     tf_weights_names, tf_biases,
                                     tf_bias_names):
    (w,) = self._cudnn_to_tf_weights(*weights)
    (b,) = self._cudnn_to_tf_biases(*biases)

    tf_weights.append(w)
    tf_weights_names.append(prefix + "/kernel")

    tf_biases.append(b)
    tf_bias_names.append(prefix + "/bias")


class CudnnGRUSaveable(CudnnOpaqueParamsSaveable):
  """SaveableObject implementation handling Cudnn GRU opaque params."""

  _rnn_mode = CUDNN_GRU
  _num_params_per_layer = CUDNN_GRU_PARAMS_PER_LAYER

  # pylint:disable=protected-access
  _rnn_cell_name = base_layer._to_snake_case(CudnnCompatibleGRUCell.__name__)

  # pylint:enable=protected-access

  def _cudnn_to_tf_weights(self, *cu_weights):
    r"""Stitching cudnn canonical weights to generate tf canonical weights."""
    w_i, w_r, w_h, r_i, r_r, r_h = cu_weights

    # pylint: disable=invalid-name
    W_i = array_ops.concat([w_i, r_i], axis=1)
    W_r = array_ops.concat([w_r, r_r], axis=1)
    # pylint: enable=invalid-name
    return (array_ops.transpose(array_ops.concat([W_i, W_r], axis=0)),
            array_ops.transpose(w_h), array_ops.transpose(r_h))

  def _tf_to_cudnn_weights(self, layer, *tf_weights):
    r"""Reverse the operations in StitchWeights()."""
    input_size = self._input_size
    num_units = self._num_units
    if layer == 0:
      input_weight_width = input_size
    else:
      input_weight_width = num_units
      if self._direction == CUDNN_RNN_BIDIRECTION:
        input_weight_width *= 2
    # pylint: disable=invalid-name
    W_ir, w_h, r_h = tf_weights
    W_ir = array_ops.transpose(W_ir)
    w_h = array_ops.transpose(w_h)
    r_h = array_ops.transpose(r_h)

    W_i, W_r = array_ops.split(W_ir, 2, axis=0)
    w_i, r_i = array_ops.split(W_i, [input_weight_width, num_units], axis=1)
    w_r, r_r = array_ops.split(W_r, [input_weight_width, num_units], axis=1)
    # pylint: enable=invalid-name
    return w_i, w_r, w_h, r_i, r_r, r_h

  def _cudnn_to_tf_biases(self, *biases):
    r"""Stitching cudnn canonical biases to generate tf canonical biases."""
    b_wi, b_wr, b_wh, b_ri, b_rr, b_rh = biases
    return (
        # Save only the sum instead of individual biases. When recovering,
        # return two biases each with half the value. Since RNN does not
        # regularize by weight decay, it has no side effect in training or
        # inference.
        array_ops.concat([b_wi, b_wr], axis=0) + array_ops.concat(
            [b_ri, b_rr], axis=0),
        b_wh,
        b_rh)

  def _tf_to_cudnn_biases(self, *tf_biases):
    r"""Reverse the operations in StitchBiases()."""
    # b_ir is the summed bias of reset and update gate.
    b_ir, b_wh, b_rh = tf_biases
    bi, br = b_ir * 0.5, b_ir * 0.5
    b_wi, b_wr = array_ops.split(bi, 2, axis=0)
    b_ri, b_rr = array_ops.split(br, 2, axis=0)
    return b_wi, b_wr, b_wh, b_ri, b_rr, b_rh

  def _TransformSingleLayerCanonical(self, weights, biases, prefix, tf_weights,
                                     tf_weights_names, tf_biases,
                                     tf_bias_names):
    # pylint: disable=invalid-name
    W_ir, w_h, r_h = self._cudnn_to_tf_weights(*weights)
    b_ir, b_wh, b_rh = self._cudnn_to_tf_biases(*biases)
    # pylint: enable=invalid-name

    tf_weights.extend([W_ir, w_h, r_h])
    tf_weights_names.append(prefix + "/gates/kernel")
    tf_weights_names.append(prefix + "/candidate/input_projection/kernel")
    tf_weights_names.append(prefix + "/candidate/hidden_projection/kernel")

    tf_biases.extend([b_ir, b_wh, b_rh])
    tf_bias_names.append(prefix + "/gates/bias")
    tf_bias_names.append(prefix + "/candidate/input_projection/bias")
    tf_bias_names.append(prefix + "/candidate/hidden_projection/bias")


class CudnnRNNSimpleSaveable(CudnnLSTMSaveable):
  """SaveableObject implementation handling Cudnn RNN Tanh opaque params."""

  # pylint:disable=protected-access
  _rnn_cell_name = base_layer._to_snake_case(
      rnn_cell_impl.BasicRNNCell.__name__)

  # pylint:enable=protected-access

  def _cudnn_to_tf_weights(self, *cu_weights):
    r"""Stitching cudnn canonical weights to generate tf canonical weights."""
    w_i, w_h = cu_weights
    W = array_ops.concat([w_i, w_h], axis=1)  # pylint: disable=invalid-name
    return (array_ops.transpose(W),)

  def _tf_to_cudnn_weights(self, layer, *tf_weights):
    r"""Reverse the operations in StitchWeights()."""
    input_size = self._input_size
    num_units = self._num_units
    if layer == 0:
      input_weight_width = input_size
    else:
      input_weight_width = num_units
      if self._direction == CUDNN_RNN_BIDIRECTION:
        input_weight_width *= 2

    (tf_weight,) = tf_weights
    # pylint: disable=invalid-name
    W = array_ops.transpose(tf_weight)
    w_i, w_h = array_ops.split(W, [input_weight_width, num_units], axis=1)
    return w_i, w_h
    # pylint: enable=invalid-name

  def _cudnn_to_tf_biases(self, *cu_biases):
    r"""Stitching cudnn canonical biases to generate tf canonical biases."""
    # Save only the sum instead of individual biases. When recovering, return
    # two biases each with half the value. Since RNN does not regularize by
    # weight decay, it has no side effect in training or inference.
    b_wi, b_wh = cu_biases
    return (b_wi + b_wh,)

  def _tf_to_cudnn_biases(self, *tf_biases):
    r"""Reverse the operations in StitchBiases()."""
    (tf_bias,) = tf_biases
    b_i = tf_bias * 0.5
    b_h = tf_bias * 0.5
    return b_i, b_h


class CudnnRNNTanhSaveable(CudnnRNNSimpleSaveable):
  """SaveableObject implementation handling Cudnn RNN Tanh opaque params."""
  _rnn_mode = CUDNN_RNN_TANH
  _num_params_per_layer = CUDNN_RNN_TANH_PARAMS_PER_LAYER


class CudnnRNNReluSaveable(CudnnRNNSimpleSaveable):
  """SaveableObject implementation handling Cudnn RNN Relu opaque params."""
  _rnn_mode = CUDNN_RNN_RELU
  _num_params_per_layer = CUDNN_RNN_RELU_PARAMS_PER_LAYER


_cudnn_rnn_common_doc_string = """
  Cudnn RNN has an opaque parameter buffer that can be used for inference and
  training. But it is possible that the layout of the parameter buffers
  changes between generations. So it is highly recommended to use
  CudnnOpaqueParamsSaveable to save and restore weights and biases in a
  canonical format.

  This is a typical use case:

    * The user creates a CudnnRNN model.
    * The user query that parameter buffer size.
    * The user creates a variable of that size that serves as the parameter
        buffers.
    * The user either initialize the parameter buffer, or load the canonical
        weights into the parameter buffer.
    * The user calls the model with the parameter buffer for inference, or
        training.
    * If training, the user creates a Saver object.
    * If training, the user creates a CudnnOpaqueParamsSaveable object from the
        parameter buffer for it to be later saved in the canonical format. When
        creating a CudnnOpaqueParamsSaveable object, a name could be provided,
        which is useful in distinguishing the names of multiple
        CudnnOpaqueParamsSaveable objects (e.g. for an encoder-decoder model).
    * Once a while, the user saves the parameter buffer into model checkpoints
        with Saver.save().
    * When restoring, the user creates a CudnnOpaqueParamsSaveable object and
      uses Saver.restore() to restore the parameter buffer from the canonical
      format to a user-defined format, as well as to restore other savable
      objects in the checkpoint file.
"""


def _check_rnn_mode(rnn_mode):
  if rnn_mode not in (CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_TANH, CUDNN_RNN_RELU):
    raise ValueError("Invalid rnn_mode: %s, expect one of (%s, %s, %s, %s)" %
                     (rnn_mode, CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_TANH,
                      CUDNN_RNN_RELU))


def _get_seed(seed):
  seed, seed2 = random_seed.get_seed(seed)
  if seed is None and seed2 is None:
    seed, seed2 = 0, 0
  return seed, seed2


def check_direction(direction):
  """Check validity of direction."""
  if direction not in (CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION):
    raise ValueError("Invalid direction: %s, expecting %s or %s" %
                     (direction, CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION))


def check_input_mode(input_mode):
  if input_mode not in (CUDNN_INPUT_LINEAR_MODE, CUDNN_INPUT_SKIP_MODE,
                        CUDNN_INPUT_AUTO_MODE):
    raise ValueError("Invalid input_mode: %s, expect one of (%s, %s, %s)" %
                     (input_mode, CUDNN_INPUT_LINEAR_MODE,
                      CUDNN_INPUT_SKIP_MODE, CUDNN_INPUT_AUTO_MODE))


def _get_num_params(rnn_mode, num_layers, direction):
  """Return num params for given Cudnn config."""
  if rnn_mode == CUDNN_LSTM:
    num_params_per_layer = CUDNN_LSTM_PARAMS_PER_LAYER
  elif rnn_mode == CUDNN_GRU:
    num_params_per_layer = CUDNN_GRU_PARAMS_PER_LAYER
  elif rnn_mode == CUDNN_RNN_RELU:
    num_params_per_layer = CUDNN_RNN_RELU_PARAMS_PER_LAYER
  elif rnn_mode == CUDNN_RNN_TANH:
    num_params_per_layer = CUDNN_RNN_TANH_PARAMS_PER_LAYER
  else:
    raise ValueError("Invalid \'rnn_mode\': %s", rnn_mode)
  num_params = num_layers * num_params_per_layer
  if direction != CUDNN_RNN_UNIDIRECTION:
    num_params *= 2
  return num_params


def _cudnn_rnn(inputs,
               input_h,
               input_c,
               params,
               is_training,
               rnn_mode,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               dropout=0.,
               seed=0,
               name=None):
  """Cudnn RNN.

  Args:
    inputs: the input sequence to the RNN model. A Tensor of shape [?,
      batch_size, input_size].
    input_h: the initial hidden state for h. A Tensor of shape [num_layers,
      batch_size, num_units].
    input_c: the initial hidden state for c. This is only relevant for LSTM.
      A Tensor of the same shape as input_h.
    params: the parameter buffer created for this model.
    is_training: whether this operation will be used in training or inference
    rnn_mode: one of ('lstm', 'gru', 'rnn_relu', 'rnn_tanh').
    input_mode: indicate whether there is a linear projection between the
      input and the actual computation before the first layer. It could be
      'linear_input', 'skip_input' or 'auto_select'.
      'linear_input' (default) always applies a linear projection of input
      onto RNN hidden state. (standard RNN behavior).
      'skip_input' is only allowed when input_size == num_units;
      'auto_select' implies 'skip_input' when input_size == num_units;
      otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    outputs, output_h, output_c
  """
  _check_rnn_mode(rnn_mode)
  check_direction(direction)
  check_input_mode(input_mode)
  seed, seed2 = random_seed.get_seed(seed)
  outputs, output_h, output_c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
      input=inputs,
      input_h=input_h,
      input_c=input_c,
      params=params,
      is_training=is_training,
      rnn_mode=rnn_mode,
      input_mode=input_mode,
      direction=direction,
      dropout=dropout,
      seed=seed,
      seed2=seed2,
      name=name)
  return (outputs, output_h, output_c)


def cudnn_lstm(inputs,
               input_h,
               input_c,
               params,
               is_training,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               dropout=0.,
               seed=0,
               name=None):
  """Cudnn LSTM.

  Args:
    inputs: the input sequence to the RNN model. A Tensor of shape [?,
      batch_size, input_size].
    input_h: the initial hidden state for h. A Tensor of shape [num_layers,
      batch_size, num_units].
    input_c: the initial hidden state for c. This is only relevant for LSTM.
      A Tensor of the same shape as input_h.
    params: the parameter buffer created for this model.
    is_training: whether this operation will be used in training or inference
      input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    outputs, output_h, output_c
  """
  return _cudnn_rnn(inputs, input_h, input_c, params, is_training, CUDNN_LSTM,
                    input_mode, direction, dropout, seed, name)


def _cudnn_rnn_no_input_c(inputs,
                          input_h,
                          params,
                          is_training,
                          rnn_mode,
                          input_mode=CUDNN_INPUT_LINEAR_MODE,
                          direction=CUDNN_RNN_UNIDIRECTION,
                          dropout=0.,
                          seed=0,
                          name=None):
  """Cudnn RNN w/o input_c.

  Args:
    inputs: the input sequence to the RNN model. A Tensor of shape [?,
      batch_size, input_size].
    input_h: the initial hidden state for h. A Tensor of shape [num_layers,
      batch_size, num_units].
    params: the parameter buffer created for this model.
    is_training: whether this operation will be used in training or inference
    rnn_mode: one of ('lstm', 'gru', 'rnn_relu', 'rnn_tanh').
    input_mode: indicate whether there is a linear projection between the
      input and the actual computation before the first layer. It could be
      'linear_input', 'skip_input' or 'auto_select'.
      'linear_input' (default) always applies a linear projection of input
      onto RNN hidden state. (standard RNN behavior).
      'skip_input' is only allowed when input_size == num_units;
      'auto_select' implies 'skip_input' when input_size == num_units;
      otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    outputs, output_h
  """
  input_c = array_ops.constant([], dtype=input_h.dtype)
  outputs, output_h, _ = _cudnn_rnn(inputs, input_h, input_c, params,
                                    is_training, rnn_mode, input_mode,
                                    direction, dropout, seed, name)
  return outputs, output_h


def cudnn_gru(inputs,
              input_h,
              params,
              is_training,
              input_mode=CUDNN_INPUT_LINEAR_MODE,
              direction=CUDNN_RNN_UNIDIRECTION,
              dropout=0.,
              seed=0,
              name=None):
  """Cudnn GRU.

  Args:
    inputs: the input sequence to the RNN model. A Tensor of shape [?,
      batch_size, input_size].
    input_h: the initial hidden state for h. A Tensor of shape [num_layers,
      batch_size, num_units].
    params: the parameter buffer created for this model.
    is_training: whether this operation will be used in training or inference
      input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    outputs, output_h
  """
  return _cudnn_rnn_no_input_c(inputs, input_h, params, is_training, CUDNN_GRU,
                               input_mode, direction, dropout, seed, name)


def cudnn_rnn_relu(inputs,
                   input_h,
                   params,
                   is_training,
                   input_mode=CUDNN_INPUT_LINEAR_MODE,
                   direction=CUDNN_RNN_UNIDIRECTION,
                   dropout=0.,
                   seed=0,
                   name=None):
  """Cudnn RNN Relu.

  Args:
    inputs: the input sequence to the RNN model. A Tensor of shape [?,
      batch_size, input_size].
    input_h: the initial hidden state for h. A Tensor of shape [num_layers,
      batch_size, num_units].
    params: the parameter buffer created for this model.
    is_training: whether this operation will be used in training or inference
      input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    outputs, output_h
  """
  return _cudnn_rnn_no_input_c(inputs, input_h, params, is_training,
                               CUDNN_RNN_RELU, input_mode, direction, dropout,
                               seed, name)


def cudnn_rnn_tanh(inputs,
                   input_h,
                   params,
                   is_training,
                   input_mode=CUDNN_INPUT_LINEAR_MODE,
                   direction=CUDNN_RNN_UNIDIRECTION,
                   dropout=0.,
                   seed=0,
                   name=None):
  """Cudnn RNN Tanh.

  Args:
    inputs: the input sequence to the RNN model. A Tensor of shape [?,
      batch_size, input_size].
    input_h: the initial hidden state for h. A Tensor of shape [num_layers,
      batch_size, num_units].
    params: the parameter buffer created for this model.
    is_training: whether this operation will be used in training or inference
      input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    outputs, output_h
  """
  return _cudnn_rnn_no_input_c(inputs, input_h, params, is_training,
                               CUDNN_RNN_TANH, input_mode, direction, dropout,
                               seed, name)


def cudnn_rnn_opaque_params_to_canonical(rnn_mode,
                                         num_layers,
                                         num_units,
                                         input_size,
                                         params,
                                         input_mode=CUDNN_INPUT_LINEAR_MODE,
                                         direction=CUDNN_RNN_UNIDIRECTION,
                                         dropout=0,
                                         seed=0,
                                         name=None):
  """Convert cudnn opaque params to canonical.

  Args:
    rnn_mode: a string specifies the mode, under which this RNN model runs.
        Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
    num_layers: the number of layers for the RNN model.
    num_units: the number of units within the RNN model.
    input_size: the size of the input, it could be different from the
        num_units.
    params: opaque cudnn params var.
    input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    weights list and bias list
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """

  _check_rnn_mode(rnn_mode)
  check_direction(direction)
  check_input_mode(input_mode)
  num_params = _get_num_params(rnn_mode, num_layers, direction)
  seed, seed2 = random_seed.get_seed(seed)
  weights, biases = gen_cudnn_rnn_ops.cudnn_rnn_params_to_canonical(
      rnn_mode=rnn_mode,
      num_layers=num_layers,
      num_units=num_units,
      input_size=input_size,
      params=params,
      input_mode=input_mode,
      direction=direction,
      dropout=dropout,
      seed=seed,
      seed2=seed2,
      num_params=num_params,
      name=name)
  return weights, biases


def cudnn_rnn_canonical_to_opaque_params(rnn_mode,
                                         num_layers,
                                         num_units,
                                         input_size,
                                         weights,
                                         biases,
                                         input_mode=CUDNN_INPUT_LINEAR_MODE,
                                         direction=CUDNN_RNN_UNIDIRECTION,
                                         dropout=0,
                                         seed=0,
                                         name=None):
  """Converts params from the canonical format to a specific format of cuDNN.

  Args:
    rnn_mode: a string specifies the mode, under which this RNN model runs.
        Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
    num_layers: the number of layers for the RNN model.
    num_units: the number of units within the RNN model.
    input_size: the size of the input, it could be different from the
        num_units.
    weights: a Tensor for weight parameters.
    biases: a Tensor for bias parameters.
    input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    an opaque Cudnn param.
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """
  _check_rnn_mode(rnn_mode)
  check_direction(direction)
  check_input_mode(input_mode)
  seed, seed2 = random_seed.get_seed(seed)
  return gen_cudnn_rnn_ops.cudnn_rnn_canonical_to_params(
      rnn_mode=rnn_mode,
      num_layers=num_layers,
      num_units=num_units,
      input_size=input_size,
      weights=weights,
      biases=biases,
      input_mode=input_mode,
      direction=direction,
      dropout=dropout,
      seed=seed,
      seed2=seed2,
      name=name)


def cudnn_rnn_opaque_params_size(rnn_mode,
                                 num_layers,
                                 num_units,
                                 input_size,
                                 input_mode=CUDNN_INPUT_LINEAR_MODE,
                                 direction=CUDNN_RNN_UNIDIRECTION,
                                 dtype=dtypes.float32,
                                 dropout=0,
                                 seed=0,
                                 name=None):
  """Returns opaque params size for specific Cudnn config.

  Args:
    rnn_mode: a string specifies the mode, under which this RNN model runs.
        Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
    num_layers: the number of layers for the RNN model.
    num_units: the number of units within the RNN model.
    input_size: the size of the input, it could be different from the
        num_units.
    input_mode: indicate whether there is a linear projection between the
        input and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'.
        'linear_input' (default) always applies a linear projection of input
        onto RNN hidden state. (standard RNN behavior).
        'skip_input' is only allowed when input_size == num_units;
        'auto_select' implies 'skip_input' when input_size == num_units;
        otherwise, it implies 'linear_input'.
    direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
    dtype: one of tf.float32 or tf.float64.
    dropout: whether to enable dropout. With it is 0, dropout is disabled.
    seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
        for behavior.
    name: name of the operation.
  Returns:
    a int, size of Cudnn opaque params.
  Raises:
    ValueError: if rnn_mode or direction is invalid.
  """
  _check_rnn_mode(rnn_mode)
  check_direction(direction)
  check_input_mode(input_mode)
  seed, seed2 = random_seed.get_seed(seed)
  return gen_cudnn_rnn_ops.cudnn_rnn_params_size(
      rnn_mode=rnn_mode,
      num_layers=num_layers,
      num_units=num_units,
      input_size=input_size,
      T=dtype,
      S=dtypes.int32,
      dropout=dropout,
      seed=seed,
      seed2=seed2,
      input_mode=input_mode,
      direction=direction,
      name=name)[0]


class _CudnnRNN(object):
  """Creates an RNN model using the underlying Cudnn implementation.

  Note that self._NUM_PARAMS_PER_LAYER is the number of parameter sets of
  weight and bias per layer. It needs to be defined in subclasses.
  """
  __doc__ += _cudnn_rnn_common_doc_string

  # TODO(jamesqin): support float16 CuDNN RNN
  def __init__(self,
               rnn_mode,
               num_layers,
               num_units,
               input_size,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               dtype=dtypes.float32,
               dropout=0.,
               seed=0):
    """Creates a CudnnRNN model from model spec.

    Args:
      rnn_mode: a string specifies the mode, under which this RNN model runs.
          Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and the actual computation before the first layer. It could be
          'linear_input', 'skip_input' or 'auto_select'.
          'linear_input' (default) always applies a linear projection of input
          onto RNN hidden state. (standard RNN behavior).
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      dtype: dtype of params, tf.float32 or tf.float64.
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
          for behavior.
    Raises:
      ValueError: if direction is invalid.
    """
    self._num_layers = num_layers
    self._num_units = num_units
    self._input_size = input_size
    self._rnn_mode = rnn_mode
    self._input_mode = input_mode
    self._direction = direction
    self._dtype = dtype
    self._dropout = dropout
    self._seed = seed

  @property
  def input_mode(self):
    return self._input_mode

  @property
  def input_size(self):
    return self._input_size

  @property
  def num_units(self):
    return self._num_units

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def rnn_mode(self):
    return self._rnn_mode

  @property
  def direction(self):
    return self._direction

  def params_size(self):
    """Calculates the size of the opaque parameter buffer needed for this model.

    Returns:
      The calculated parameter buffer size.
    """
    return cudnn_rnn_opaque_params_size(
        rnn_mode=self._rnn_mode,
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        dtype=self._dtype,
        dropout=self._dropout,
        seed=self._seed,
        input_mode=self._input_mode,
        direction=self._direction)

  def __call__(self, input_data, input_h, input_c, params, is_training=True):
    """Runs the forward step for the RNN model.

    Args:
      input_data: the input sequence to the RNN model. A Tensor of shape [?,
        batch_size, input_size].
      input_h: the initial hidden state for h. A Tensor of shape [num_layers,
        batch_size, num_units].
      input_c: the initial hidden state for c. This is only relevant for LSTM.
        A Tensor of the same shape as input_h.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.
    Returns:
      output: the output sequence.
      output_h: the final state for h.
      output_c: the final state for c. This is only relevant for LSTM.
    """
    return _cudnn_rnn(
        input_data,
        input_h,
        input_c,
        params,
        is_training,
        self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed)

  def params_to_canonical(self, params):
    """Converts params from a specific format of cuDNN to the canonical format.

    Args:
      params: a Variable for weight and bias parameters.

    Returns:
      A function for the specific-to-canonical conversion.
    """
    return cudnn_rnn_opaque_params_to_canonical(
        rnn_mode=self._rnn_mode,
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        params=params,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed)

  def canonical_to_params(self, weights, biases):
    """Converts params from the canonical format to a specific format of cuDNN.

    Args:
      weights: a Tensor for weight parameters.
      biases: a Tensor for bias parameters.

    Returns:
      A function for the canonical-to-params-to-specific conversion..
    """
    return cudnn_rnn_canonical_to_opaque_params(
        rnn_mode=self._rnn_mode,
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        weights=weights,
        biases=biases,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed)


class CudnnLSTM(_CudnnRNN):
  """Cudnn implementation of the LSTM model."""
  __doc__ += _cudnn_rnn_common_doc_string
  # 4 sets of weight and bias parameters for the recurrent input, and 4 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = CUDNN_LSTM_PARAMS_PER_LAYER

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               dtype=dtypes.float32,
               dropout=0.,
               seed=0):
    """Creates a Cudnn LSTM model from model spec.

    Args:
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and The actual computation before the first layer. It could be
          'skip_input', 'linear_input' or 'auto_select'.
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      dtype: dtype of params, tf.float32 or tf.float64.
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the seed used for initializing dropout.
    """
    super(CudnnLSTM, self).__init__(
        CUDNN_LSTM,
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
        dtype=dtype,
        dropout=dropout,
        seed=seed)

  def __call__(self, input_data, input_h, input_c, params, is_training=True):
    """Runs the forward step for the Cudnn LSTM model.

    Args:
      input_data: the input sequence to the LSTM model. A Tensor of shape [?,
        batch_size, input_size].
      input_h: the initial hidden state for h. A Tensor of shape [num_layers,
        batch_size, num_units].
      input_c: the initial hidden state for c. A Tensor of the same shape as
        input_h.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.
    Returns:
      output: the output sequence.
      output_h: the final state for h.
      output_c: the final state for c.
    """
    output, output_h, output_c = super(CudnnLSTM, self).__call__(
        input_data, input_h, input_c, params, is_training=is_training)
    return (output, output_h, output_c)


class _CudnnRNNNoInputC(_CudnnRNN):
  """Simple CudnnRNN models without input_c."""
  __doc__ += _cudnn_rnn_common_doc_string

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               dtype=dtypes.float32,
               dropout=0.,
               seed=0):
    """Creates a Cudnn RNN model from model without hidden-state C.

    Args:
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and The actual computation before the first layer. It could be
          'skip_input', 'linear_input' or 'auto_select'.
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      dtype: dtype of params, tf.float32 or tf.float64.
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the seed used for initializing dropout.

    Raises:
      ValueError: if direction is not 'unidirectional' or 'bidirectional'.
    """

    if direction not in (CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION):
      raise ValueError("Invalid direction: %s", direction)

    super(_CudnnRNNNoInputC, self).__init__(
        self._rnn_mode,
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
        dtype=dtype,
        dropout=dropout,
        seed=seed)

  def __call__(self, input_data, input_h, params, is_training=True):
    """Runs the forward step for the Cudnn LSTM model.

    Args:
      input_data: the input sequence to the RNN model. A Tensor of shape [?,
        batch_size, input_size].
      input_h: the initial hidden state for h. A Tensor of shape [num_layers,
        batch_size, num_units].
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.
    Returns:
      output: the output sequence.
      output_h: the final state for h.
    """
    return _cudnn_rnn_no_input_c(
        input_data,
        input_h,
        params,
        is_training,
        self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed)


class CudnnGRU(_CudnnRNNNoInputC):
  """Cudnn implementation of the GRU model."""
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = CUDNN_GRU
  # 3 sets of weight and bias parameters for the recurrent input, and 3 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = CUDNN_GRU_PARAMS_PER_LAYER


class CudnnRNNTanh(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-tanh model."""
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = CUDNN_RNN_TANH
  # 1 set of weight and bias parameters for the recurrent input, and 1 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = CUDNN_RNN_TANH_PARAMS_PER_LAYER


class CudnnRNNRelu(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-relu model."""
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = CUDNN_RNN_RELU
  # 1 set of weight and bias parameters for the recurrent input, and 1 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = CUDNN_RNN_RELU_PARAMS_PER_LAYER


@ops.RegisterGradient("CudnnRNN")
def _cudnn_rnn_backward(op, *grad):
  if not op.get_attr("is_training"):
    raise ValueError(
        "CudnnRNN must set is_training to True to be used in gradients")
  return gen_cudnn_rnn_ops.cudnn_rnn_backprop(
      input=op.inputs[0],
      input_h=op.inputs[1],
      input_c=op.inputs[2],
      params=op.inputs[3],
      output=op.outputs[0],
      output_h=op.outputs[1],
      output_c=op.outputs[2],
      output_backprop=grad[0],
      output_h_backprop=grad[1],
      output_c_backprop=grad[2],
      reserve_space=op.outputs[3],
      dropout=op.get_attr("dropout"),
      seed=op.get_attr("seed"),
      seed2=op.get_attr("seed2"),
      rnn_mode=op.get_attr("rnn_mode"),
      input_mode=op.get_attr("input_mode"),
      direction=op.get_attr("direction"))


ops.RegisterShape("CudnnRNNParamsSize")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("CudnnRNNParamsToCanonical")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("CudnnRNNCanonicalToParams")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("CudnnRNN")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("CudnnRNNBackprop")(common_shapes.call_cpp_shape_fn)
