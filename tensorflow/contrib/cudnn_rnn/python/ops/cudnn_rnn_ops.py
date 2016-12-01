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
from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import resource_loader

_cudnn_rnn_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_cudnn_rnn_ops.so"))

_cudnn_rnn_common_doc_string = """
  Cudnn RNN has an opaque parameter buffer that can be used for inference and
  training. But it is possible that the layout of the parameter buffers
  changes between generations. So it is highly recommended to use the canonical
  weights and biases to for saving and restoring a model.

  This is a typical use case:
    * The user creates a CudnnRNN model.
    * The user query that parameter buffer size.
    * The user creates a variable of that size that serves as the parameter
        buffers.
    * The user either initialize the parameter buffer, or load the canonical
        weights into the parameter buffer.
    * The user calls the model with the parameter buffer for inference, or
        training.
    * Once a while, the user extracts the canonical weights from the parameter
        buffer and saves them into model checkpoints.
"""


class _CudnnRNN(object):
  """Create an RNN model using the underlying Cudnn implementation.
  """
  __doc__ += _cudnn_rnn_common_doc_string

  def __init__(self,
               rnn_mode,
               num_layers,
               num_units,
               input_size,
               input_mode="auto_select",
               direction="unidirectional",
               dropout=0.,
               seed=0,
               seed2=0):
    """Create a CudnnRNN model from model spec.

    Args:
      rnn_mode: a string specifies the mode, under which this RNN model runs.
          Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
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
      seed: the first part of a seed that is used to initialize dropout.
      seed2: the second part of a seed that is used to initialize dropout.
    """
    self._num_layers = num_layers
    self._num_units = num_units
    self._input_size = input_size
    self._rnn_mode = rnn_mode
    self._input_mode = input_mode
    self._direction = direction
    self._dropout = dropout
    self._seed = seed
    self._seed2 = seed2

  def params_size(self):
    """Calculate the size of the opaque parameter buffer needed for this model.

    Returns:
      The calculated parameter buffer size.
    """
    return gen_cudnn_rnn_ops.cudnn_rnn_params_size(
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        T=dtypes.float32,
        S=dtypes.int32,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction)[0]

  def __call__(self, input_data, input_h, input_c, params, is_training=True):
    """Run the forward step for the RNN model.

    Args:
      input_data: the input sequence to the RNN model.
      input_h: the initial hidden state for h.
      input_c: the initial hidden state for c. This is only relevant for LSTM.
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

  # TODO(zhengxq): add reading and writing canonical weights.


class CudnnLSTM(_CudnnRNN):
  """Cudnn implementation of the LSTM model.
  """
  __doc__ += _cudnn_rnn_common_doc_string

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode="auto_select",
               direction="unidirectional",
               dropout=0.,
               seed=0,
               seed2=0):
    """Create a Cudnn LSTM model from model spec.

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
      seed: the first part of a seed that is used to initialize dropout.
      seed2: the second part of a seed that is used to initialize dropout.
    """
    super(CudnnLSTM, self).__init__(
        "lstm",
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
        dropout=dropout,
        seed=seed,
        seed2=seed2)

  def __call__(self, input_data, input_h, input_c, params, is_training=True):
    """Run the forward step for the Cudnn LSTM model.

    Args:
      input_data: the input sequence to the LSTM model.
      input_h: the initial hidden state for h.
      input_c: the initial hidden state for c.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.

    Returns:
      output: the output sequuence.
      output_h: the final state for h.
      output_c: the final state for c.
    """
    output, output_h, output_c = super(CudnnLSTM, self).__call__(input_data,
                                                                 input_h,
                                                                 input_c,
                                                                 params,
                                                                 is_training)
    return (output, output_h, output_c)


class _CudnnRNNNoInputC(_CudnnRNN):
  """Simple CudnnRNN models without input_c.
  """
  __doc__ += _cudnn_rnn_common_doc_string

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode="auto_select",
               direction="unidirectional",
               dropout=0.,
               seed=0,
               seed2=0):
    """Create a Cudnn RNN model from model without hidden-state C.

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
      seed: the first part of a seed that is used to initialize dropout.
      seed2: the second part of a seed that is used to initialize dropout.
    """
    super(_CudnnRNNNoInputC, self).__init__(
        self._rnn_mode,
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
        dropout=dropout,
        seed=seed,
        seed2=seed2)

  def __call__(self, input_data, input_h, params, is_training=True):
    """Run the forward step for the Cudnn LSTM model.

    Args:
      input_data: the input sequence to the LSTM model.
      input_h: the initial hidden state for h.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.

    Returns:
      output: the output sequuence.
      output_h: the final state for h.
    """
    output, output_h, _ = super(_CudnnRNNNoInputC, self).__call__(
        input_data, input_h, None, params, is_training=True)
    return (output, output_h)


class CudnnGRU(_CudnnRNNNoInputC):
  """Cudnn implementation of the GRU model.
  """
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = "gru"


class CudnnRNNTanh(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-tanh model.
  """
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = "rnn_tanh"


class CudnnRNNRelu(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-relu model.
  """
  __doc__ += _cudnn_rnn_common_doc_string
  _rnn_mode = "rnn_relu"


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
      rnn_mode=op.get_attr("rnn_mode"),
      input_mode=op.get_attr("input_mode"),
      direction=op.get_attr("direction"))
