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

from tensorflow.contrib.cudnn_rnn.ops import gen_cudnn_rnn_ops
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.training import saver

_cudnn_rnn_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_cudnn_rnn_ops.so"))


# TODO(yaozhang): make sure we only save the canonical version of params and
# don't save the platform-specific version to avoid potential race
# conditions where params is updated by both versions when being restored.
# Currently, checkpointing will function properly, despite that we save both
# versions, because Saver restores customized savables after Variables.
# However, it is good to not rely on this restoring order of Saver and to
# avoid unnecessary storage. Add a test to check only the canonical version is
# saved.
class RNNParamsSaveable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation that handles the RNN params variable."""

  def __init__(self,
               cudnn_rnn,
               params_to_canonical,
               canonical_to_params,
               param_variables,
               base_variable_scope=None,
               name="params_canonical"):
    """Creates a RNNParamsSaveable object.

       RNNParamsSaveable is saveable/restorable in a checkpoint file and is used
       to save/restore the weights and biases parameters in a canonical
       format, where parameters are saved as tensors layer by layer. For each
       layer, the bias tensors are saved following the weight tensors. When
       restoring, a user could name param_variables as desired, and restore
       weight and bias tensors to these variables.

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
      cudnn_rnn: cudnn RNN class instance.
      params_to_canonical: a function to convert params from a specific format
          for cuDNN or other RNN ops to the canonical format.
          _CudnnRNN.params_to_canonical() should be provided here.
      canonical_to_params: a function to convert params from the canonical
          format to a specific format for cuDNN or other RNN ops. The function
          must return a scalar (e.g. in the case of cuDNN) or a tuple. This
          function could be _CudnnRNN.canonical_to_params() or a
          user-defined function.
      param_variables: a list of Variables for parameters in a specific form.
          For cuDNN RNN ops, this is a single merged variable for both weights
          and biases; for other RNN ops, this might be multiple unmerged or
          partially merged variables respectively for weights and biases.
      base_variable_scope: a string, name of outer variable scope, used as
          part of prefix of names of saved variables.
      name: the name of the RNNParamsSaveable object.
    """
    # There is only a single merged parameter variable for cuDNN when saving.
    self._cudnn_rnn = cudnn_rnn
    weights, biases = params_to_canonical(param_variables[0])
    weights, biases, = self._transform_canonical(weights, biases)
    weight_names, biase_names = self._transformed_canonical_names(
        weights, biases)
    self._canonical_to_params = canonical_to_params
    self._variables = param_variables
    # We currently don't use slice_spec. It might be useful in a distributed
    # setting where each parameter server node stores a slice of variable,
    # instead of having the master pull all slices and then save them.
    slice_spec = ""
    params = weights + biases
    param_names = weight_names + biase_names
    if base_variable_scope:
      param_names = ["%s/%s" % (base_variable_scope, pn) for pn in param_names]
    specs = [
        saver.BaseSaverBuilder.SaveSpec(param, slice_spec, param_name)
        for param, param_name in zip(params, param_names)
    ]
    super(RNNParamsSaveable, self).__init__(None, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    if (self._cudnn_rnn.direction == "unidirectional" and
        self._cudnn_rnn.rnn_mode == "lstm"):
      assert len(restored_tensors) % 4 == 0
      weights = restored_tensors[:len(restored_tensors) // 4]
      biases = restored_tensors[len(restored_tensors) // 4:]
    else:
      weights = restored_tensors[:len(restored_tensors) // 2]
      biases = restored_tensors[len(restored_tensors) // 2:]
    weights, biases = self._untransform_canonical(weights, biases)
    params = self._canonical_to_params(weights, biases)
    if not isinstance(params, tuple):
      params = (params,)
    assign_ops = [
        state_ops.assign(variable, param, validate_shape=False)
        for variable, param in zip(self._variables, params)
    ]
    return control_flow_ops.group(*assign_ops)

  def _switch_inner(self, array, base_idx):
    array[base_idx + 1], array[base_idx + 2] = (array[base_idx + 2],
                                                array[base_idx + 1])

  def _transform_canonical(self, weights, biases):
    if (self._cudnn_rnn.direction != "unidirectional" or
        self._cudnn_rnn.rnn_mode != "lstm"):
      return weights, biases
    return self._transform_lstm_canonical(weights, biases)

  def _transformed_canonical_names(self, weights, biases):
    """Return canonical names for transformed weight and bias tensors."""
    if (self._cudnn_rnn.direction != "unidirectional" or
        self._cudnn_rnn.rnn_mode != "lstm"):
      assert len(weights) == len(biases)
      return ([w.name for w in weights], [b.name for b in biases])
    else:
      w_names, b_names = [], []
      assert len(weights) * 3 == len(biases)
      num_layers = self._cudnn_rnn.num_layers
      # TODO(jamesqin): get rid of multi_rnn_cell when num_layers is 1
      for i in range(num_layers):
        # One transformed weight tensor each layer.
        w_names.append(
            "multi_rnn_cell/cell_%d/cudnn_compatible_lstm_cell/kernel" % i)
        # Three transformed bias tensors each layer:
        # the 1st is for CudnnCompatbleLSTM(Block)Cell restore; the latter two
        # sum up to the 1st, and are used for cuDNN restore.
        b_names.append(
            "multi_rnn_cell/cell_%d/cudnn_compatible_lstm_cell/bias" % i)
        b_names.extend([
            "multi_rnn_cell/cell_%d/cudnn_compatible_lstm_cell/bias_cudnn_%d" %
            (i, j) for j in range(2)
        ])
      return w_names, b_names

  def _transform_lstm_canonical(self, weights, biases):
    """Create transformed lstm canonical params.

    Produce properly-shaped monolithic weight and bias tensors to share between
    cuDNN and cudnn_compatible non-platform specific LSTM cells.
    Args:
      weights: a list of Tensors recovered from cuDNN params_to_canonical.
      biases: a list of Tensors recovered from cuDNN params_to_canonical.
    Returns:
      Two lists of tensors, one for weight and bias each.
      The weight list contains num_layers tensors and bias one contains 3 *
      num_layers tensors. Both original and combined biases since cuDNN biases
      are not restorable from the transformed version.
    """
    transformed_weights, transformed_biases = [], []
    for i in range(self._cudnn_rnn.num_layers):
      base_idx = i * 8
      num_units = self._cudnn_rnn.num_units
      input_size = self._cudnn_rnn.input_size if i == 0 else num_units
      # cuDNN tensor shapes per time_step:
      # input.shape:         [batch_size, input_size],
      # input_weights.shape: [num_units, input_size] (first layer)
      #                      [num_units, num_units]  (other layers)
      # state_weights.shape: [num_units, num_units]
      # biases.shape:        [num_units]
      #
      # General LSTM cells compute gate functions using:
      #   [x, h_prev] * weights + biases
      # Therefore for each layer, they expect
      # weight.shape: [input_size + num_units, 4 * num_units] (first_layer)
      #               [num_units + num_units, 4 * num_units]  (other layers)
      # bias.shape:   [4 * num_units]

      # Stitch weights together in this layer.
      stitched_w = []
      for j in range(4):
        stitched_w.append(
            array_ops.concat(
                [
                    array_ops.reshape(weights[base_idx + j],
                                      [num_units, input_size]),
                    array_ops.reshape(weights[base_idx + j + 4],
                                      [num_units, num_units])
                ],
                axis=1))
      # cuDNN weights are in ifco order, convert to icfo order.
      self._switch_inner(stitched_w, 0)
      transformed_weights.append(
          array_ops.transpose(array_ops.concat(stitched_w, axis=0)))

      # Stitch biases together in this layer.
      # Convert to icfo order.
      self._switch_inner(biases, base_idx)
      self._switch_inner(biases, base_idx + 4)
      # The bias for layer input.
      b_in = array_ops.concat(biases[base_idx:base_idx + 4], axis=0)
      # The bias for recurrent input.
      b_rec = array_ops.concat(biases[base_idx + 4:base_idx + 8], axis=0)

      transformed_biases.extend([b_in + b_rec, b_in, b_rec])
    return transformed_weights, transformed_biases

  def _untransform_canonical(self, transformed_weights, transformed_biases):
    if (self._cudnn_rnn.direction != "unidirectional" or
        self._cudnn_rnn.rnn_mode != "lstm"):
      return transformed_weights, transformed_biases
    return self._untransform_lstm_canonical(transformed_weights,
                                            transformed_biases)

  def _untransform_lstm_canonical(self, transformed_weights,
                                  transformed_biases):
    """The reverse procedure of _transform_lstm_canonical().

    Args:
      transformed_weights: a list of tensors, one for each layer.
      transformed_biases: a list of tensors , 3 for each layer: the 2nd for
        layer input, the 3rd for recurrent input, the 1st is the sum of the
        latter two.
    Returns:
      Two lists of tensors for weights and biases respectively.
      There are 8 tensors per weight and per bias for each layer:
      tensor 0-3 are applied to the input from the previous layer;
      tensor 4-7 to the recurrent input. Tensor 0 and 4 are for the input gate;
      tensor 1 and 5 the forget gate; tensor 2 and 6 the new memory gate;
      tensor 3 and 7 the output gate.
    """
    weights, biases = [], []
    assert 3 * len(transformed_weights) == len(transformed_biases)
    for i in range(len(transformed_weights)):
      num_units = self._cudnn_rnn.num_units
      input_size = self._cudnn_rnn.input_size if i == 0 else num_units
      # weights applied on layer inputs.
      wi = array_ops.slice(transformed_weights[i], [0, 0],
                           [input_size, 4 * num_units])
      # weights applied on recurrent inputs.
      wr = array_ops.slice(transformed_weights[i], [input_size, 0],
                           [num_units, 4 * num_units])
      wi_list = array_ops.split(wi, 4, axis=1)
      wr_list = array_ops.split(wr, 4, axis=1)

      for j in range(len(wi_list)):
        wi_list[j] = array_ops.reshape(array_ops.transpose(wi_list[j]), [-1])
        wr_list[j] = array_ops.reshape(array_ops.transpose(wr_list[j]), [-1])
      # canonical weights are in icfo order, convert to ifco order for cuDNN.
      self._switch_inner(wi_list, 0)
      self._switch_inner(wr_list, 0)
      weights.extend(wi_list)
      weights.extend(wr_list)

      base_idx = 3 * i
      bi_list = array_ops.split(transformed_biases[base_idx + 1], 4, axis=0)
      br_list = array_ops.split(transformed_biases[base_idx + 2], 4, axis=0)
      # canonical weights are in icfo order, convert to ifco order for cuDNN.
      self._switch_inner(bi_list, 0)
      self._switch_inner(br_list, 0)
      biases.extend(bi_list)
      biases.extend(br_list)
    return weights, biases


_cudnn_rnn_common_doc_string = """
  Cudnn RNN has an opaque parameter buffer that can be used for inference and
  training. But it is possible that the layout of the parameter buffers
  changes between generations. So it is highly recommended to use
  RNNParamsSaveable to save and restore weights and biases in a canonical
  format.

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
    * If training, the user creates a RNNParamsSaveable object from the
        parameter buffer for it to be later saved in the canonical format. When
        creating a RNNParamsSaveable object, a name could be provided, which is
        useful in distinguishing the names of multiple RNNParamsSaveable
        objects (e.g. for an encoder-decoder model).
    * Once a while, the user saves the parameter buffer into model checkpoints
        with Saver.save().
    * When restoring, the user creates a RNNParamsSaveable object and uses
      Saver.restore() to restore the parameter buffer from the canonical format
      to a user-defined format, as well as to restore other savable objects
      in the checkpoint file.
"""


class _CudnnRNN(object):
  """Creates an RNN model using the underlying Cudnn implementation.

  Note that self._NUM_PARAMS_PER_LAYER is the number of parameter sets of
  weight and bias per layer. It needs to be defined in subclasses.
  """
  __doc__ += _cudnn_rnn_common_doc_string

  def __init__(self,
               rnn_mode,
               num_layers,
               num_units,
               input_size,
               input_mode="linear_input",
               direction="unidirectional",
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
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
          for behavior.
    """
    self._num_layers = num_layers
    self._num_units = num_units
    self._input_size = input_size
    self._rnn_mode = rnn_mode
    self._input_mode = input_mode
    self._direction = direction
    self._dropout = dropout
    # get graph and op seed.
    self._seed, self._seed2 = random_seed.get_seed(seed)
    if self._seed is None and self._seed2 is None:
      self._seed, self._seed2 = 0, 0

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
    return gen_cudnn_rnn_ops.cudnn_rnn_params_size(
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        T=dtypes.float32,
        S=dtypes.int32,
        dropout=self._dropout,
        seed=self._seed,
        seed2=self._seed2,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction)[0]

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
      output: the output sequuence.
      output_h: the final state for h.
      output_c: the final state for c. This is only relevant for LSTM.
    """
    if self._rnn_mode != "lstm":
      # For model that doesn't take input_c, replace with a dummy tensor.
      input_c = array_ops.constant([], dtype=dtypes.float32)
    output, output_h, output_c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        input=input_data,
        input_h=input_h,
        input_c=input_c,
        params=params,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed,
        seed2=self._seed2,
        is_training=is_training)
    return (output, output_h, output_c)

  def params_to_canonical(self, params):
    """Converts params from a specific format of cuDNN to the canonical format.

    Args:
      params: a Variable for weight and bias parameters.

    Returns:
      A function for the specific-to-canonical conversion.
    """
    weights, biases = gen_cudnn_rnn_ops.cudnn_rnn_params_to_canonical(
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        params=params,
        dropout=self._dropout,
        seed=self._seed,
        seed2=self._seed2,
        num_params=self._num_layers * self._NUM_PARAMS_PER_LAYER,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction)
    return weights, biases

  def canonical_to_params(self, weights, biases):
    """Converts params from the canonical format to a specific format of cuDNN.

    Args:
      weights: a Tensor for weight parameters.
      biases: a Tensor for bias parameters.

    Returns:
      A function for the canonical-to-params-to-specific conversion..
    """
    return gen_cudnn_rnn_ops.cudnn_rnn_canonical_to_params(
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        weights=weights,
        biases=biases,
        dropout=self._dropout,
        seed=self._seed,
        seed2=self._seed2,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction)


class CudnnLSTM(_CudnnRNN):
  """Cudnn implementation of the LSTM model."""
  __doc__ += _cudnn_rnn_common_doc_string
  # 4 sets of weight and bias parameters for the recurrent input, and 4 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 8

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode="linear_input",
               direction="unidirectional",
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
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the seed used for initializing dropout.
    """
    super(CudnnLSTM, self).__init__(
        "lstm",
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
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
      output: the output sequuence.
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
               input_mode="linear_input",
               direction="unidirectional",
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
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the seed used for initializing dropout.
    """

    super(_CudnnRNNNoInputC, self).__init__(
        self._rnn_mode,
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
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
      output: the output sequuence.
      output_h: the final state for h.
    """
    output, output_h, _ = super(_CudnnRNNNoInputC, self).__call__(
        input_data, input_h, None, params, is_training=is_training)
    return (output, output_h)


class CudnnGRU(_CudnnRNNNoInputC):
  """Cudnn implementation of the GRU model."""
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = "gru"
  # 3 sets of weight and bias parameters for the recurrent input, and 3 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 6


class CudnnRNNTanh(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-tanh model."""
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = "rnn_tanh"
  # 1 set of weight and bias parameters for the recurrent input, and 1 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 2


class CudnnRNNRelu(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-relu model."""
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = "rnn_relu"
  # 1 set of weight and bias parameters for the recurrent input, and 1 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 2


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


class CudnnCompatibleLSTMCell(rnn_cell_impl.LSTMCell):
  """Cudnn Compatible LSTMCell.

  A simple wrapper around @{tf.nn.rnn_cell.LSTMCell} to use along with
  @{tf.contrib.cudnn_rnn.CudnnLSTM}. The latter's params can be used by the
  former seamlessly.
  """

  def __init__(self, num_units, reuse=None):
    super(CudnnCompatibleLSTMCell, self).__init__(
        num_units,
        use_peepholes=False,
        cell_clip=None,
        num_proj=None,
        proj_clip=None,
        state_is_tuple=True,
        activation=None,
        reuse=reuse,
        forget_bias=0)


class CudnnCompatibleLSTMBlockCell(lstm_ops.LSTMBlockCell):
  """Cudnn Compatible LSTMBlockCell.

  A simple wrapper around @{tf.contrib.rnn.LSTMBlockCell} to use along with
  @{tf.contrib.cudnn_rnn.CudnnLSTM}. The latter's params can be used by the
  former seamlessly. It is the more performant than
  ${tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell}, the same way
  @{tf.contrib.rnn.LSTMBlockCell} is than @{tf.nn.rnn_cell.LSTMCell}.
  """

  def __init__(self, num_units):
    super(CudnnCompatibleLSTMBlockCell, self).__init__(
        num_units, forget_bias=0, clip_cell=False, use_peephole=False)
    self._names.update({"scope": "cudnn_compatible_lstm_cell"})
