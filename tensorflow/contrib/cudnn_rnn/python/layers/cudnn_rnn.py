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

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging


CUDNN_RNN_UNIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
CUDNN_RNN_BIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION
CUDNN_LSTM = cudnn_rnn_ops.CUDNN_LSTM
CUDNN_GRU = cudnn_rnn_ops.CUDNN_GRU
CUDNN_RNN_RELU = cudnn_rnn_ops.CUDNN_RNN_RELU
CUDNN_RNN_TANH = cudnn_rnn_ops.CUDNN_RNN_TANH

# Half for cell input, half for hidden states.
CUDNN_LSTM_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_LSTM_PARAMS_PER_LAYER
CUDNN_GRU_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_GRU_PARAMS_PER_LAYER
CUDNN_RNN_TANH_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_RNN_TANH_PARAMS_PER_LAYER
CUDNN_RNN_RELU_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_RNN_RELU_PARAMS_PER_LAYER

CUDNN_INPUT_LINEAR_MODE = cudnn_rnn_ops.CUDNN_INPUT_LINEAR_MODE
CUDNN_INPUT_SKIP_MODE = cudnn_rnn_ops.CUDNN_INPUT_SKIP_MODE
CUDNN_INPUT_AUTO_MODE = cudnn_rnn_ops.CUDNN_INPUT_AUTO_MODE


__all__ = ["CudnnLSTM", "CudnnGRU", "CudnnRNNTanh", "CudnnRNNRelu"]


class _CudnnRNN(base_layer.Layer):
  # pylint:disable=line-too-long
  """Abstract class for RNN layers with Cudnn implementation.

  Cudnn RNNs have two major differences from other platform-independent RNNs tf
  provides:
  * Cudnn LSTM and GRU are mathematically different from their tf counterparts.
    (e.g. @{tf.contrib.rnn.LSTMBlockCell} and @{tf.nn.rnn_cell.GRUCell}.
  * Cudnn-trained checkpoints are not directly compatible with tf RNNs:
    * They use a single opaque parameter buffer for the entire (possibly)
      multi-layer multi-directional RNN; Whereas tf RNN weights are per-cell and
      layer.
    * The size and layout of the parameter buffers may change between
      CUDA/CuDNN/GPU generations. Because of that, the opaque parameter variable
      does not have a static shape and is not partitionable. Instead of using
      partitioning to alleviate the PS's traffic load, try building a
      multi-tower model and do gradient aggregation locally within the host
      before updating the PS. See https://www.tensorflow.org/performance/performance_models#parameter_server_variables
      for a detailed performance guide.

  Consequently, if one plans to use Cudnn trained models on both GPU and CPU
  for inference and training, one needs to:
  * Create a CudnnOpaqueParamsSaveable subclass object to save RNN params in
    canonical format. (This is done for you automatically during layer building
    process.)
  * When not using a Cudnn RNN class, use CudnnCompatibleRNN classes to load the
    checkpoints. These classes are platform-independent and perform the same
    computation as Cudnn for training and inference.
  Similarly, CudnnCompatibleRNN-trained checkpoints can be loaded by CudnnRNN
  classes seamlessly.

  Below is a typical workflow(using LSTM as an example):
  for detailed performance guide.

  # Use Cudnn-trained checkpoints with CudnnCompatibleRNNs
  ```python
  with tf.Graph().as_default():
    lstm = CudnnLSTM(num_layers, num_units, direction, ...)

    outputs, output_states = lstm(inputs, initial_states, training=True)

    # If user plans to delay calling the cell with inputs, one can do
    # lstm.build(input_shape)

    saver = Saver()

    # training subgraph
    ...

    # Once in a while save the model.
    saver.save(save_path)

  # Inference subgraph for unidirectional RNN on, e.g., CPU or mobile.
  with tf.Graph().as_default():
    single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTM(num_units)

    # NOTE: Even if there's only one layer, the cell needs to be wrapped in
    # MultiRNNCell.
    cell = tf.nn.rnn_cell.MultiRNNCell(
      [single_cell() for _ in range(num_layers)])

    # Leave the scope arg unset.
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state, ...)

    saver = Saver()

    # Create session
    sess = ...

    # Restores
    saver.restore(sess, save_path)

  # Inference subgraph for bidirectional RNN
  with tf.Graph().as_default():
    single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTM(num_units)
    cells_fw = [single_cell() for _ in range(num_layers)]
    cells_bw = [single_cell() for _ in range(num_layers)]

    # Leave the scope arg unset.
    (outputs, output_state_fw,
     output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
         cells_fw, cells_bw, inputs, ...)
    saver = Saver()

    # Create session
    sess = ...

    # Restores
    saver.restore(sess, save_path)
  ```
  """
  # pylint:enable=line-too-long

  # The following are constants defined by subclasses.
  # Type of RNN cell.
  _rnn_mode = None
  # Number of cell weights(or biases) per layer.
  _num_params_per_layer = None
  # Custom SaveableObject class for the CudnnRNN class.
  _saveable_cls = None

  def __init__(self,
               num_layers,
               num_units,
               input_mode=CUDNN_INPUT_LINEAR_MODE,
               direction=CUDNN_RNN_UNIDIRECTION,
               dropout=0.,
               seed=None,
               dtype=dtypes.float32,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    """Creates a CudnnRNN model from model spec.

    Args:
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_mode: indicate whether there is a linear projection between the
          input and the actual computation before the first layer. It can be
          'linear_input', 'skip_input' or 'auto_select'.
          'linear_input' (default) always applies a linear projection of input
          onto RNN hidden state. (standard RNN behavior).
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Can be either
          'unidirectional' or 'bidirectional'
      dropout: dropout rate, a number between [0, 1]. Dropout is applied on
          inputs of each layer. When set to 0, dropout is disabled.
      seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
          for behavior.
      dtype: tf.float16, tf.float32 or tf.float64
      kernel_initializer: starting value to initialize the weight.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking __call__().

    Raises:
      ValueError: if direction is invalid. Or dtype is not supported.
    """
    super(_CudnnRNN, self).__init__(dtype=dtype, name=name)
    cudnn_rnn_ops.check_direction(direction)
    cudnn_rnn_ops.check_input_mode(input_mode)

    if dtype not in [dtypes.float16, dtypes.float32, dtypes.float64]:
      raise ValueError(
          "Only support float16, float32, float64, provided %s" % dtype)
    # Layer self.dtype is type name, the original DType object is kept here.
    self._plain_dtype = dtype
    self._num_layers = num_layers
    self._num_units = num_units
    self._input_mode = input_mode
    self._direction = direction
    self._dropout = dropout
    self._seed = seed
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    # Init input_size to None, which will be set after build().
    self._input_size = None
    self._saveable = None

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def num_units(self):
    return self._num_units

  @property
  def input_mode(self):
    """Input mode of first layer.

    Indicates whether there is a linear projection between the input and the
    actual computation before the first layer. It can be
    * 'linear_input': (default) always applies a linear projection of input
      onto RNN hidden state. (standard RNN behavior)
    * 'skip_input': 'skip_input' is only allowed when input_size == num_units.
    * 'auto_select'. implies 'skip_input' when input_size == num_units;
      otherwise, it implies 'linear_input'.

    Returns:
      'linear_input', 'skip_input' or 'auto_select'.
    """
    return self._input_mode

  @property
  def input_size(self):
    if not self._input_size:
      raise ValueError(
          "\'input_size\' is unknown since layer has not been built.")
    return self._input_size

  @property
  def rnn_mode(self):
    """Type of RNN cell used.

    Returns:
      `lstm`, `gru`, `rnn_relu` or `rnn_tanh`.
    """
    return self._rnn_mode

  @property
  def direction(self):
    """Returns `unidirectional` or `bidirectional`."""
    return self._direction

  @property
  def num_dirs(self):
    return 1 if self._direction == CUDNN_RNN_UNIDIRECTION else 2

  @property
  def saveable(self):
    return self._saveable

  @property
  def canonical_weight_shapes(self):
    """Shapes of Cudnn canonical weight tensors."""
    if not self._input_size:
      raise RuntimeError(
          "%s.canonical_weight_shapes invoked before input shape is known" %
          type(self).__name__)

    shapes = []
    for i in range(self._num_layers):
      shapes.extend(self._canonical_weight_shape(i))
    return shapes

  @property
  def canonical_bias_shapes(self):
    """Shapes of Cudnn canonical bias tensors."""
    return self._canonical_bias_shape(0) * self._num_layers

  def _update_trainable_weights(self, getter, *args, **kwargs):
    """Custom getter for layer variables."""
    # Add variables to layer's `(non_)trainable_weights` list(s).
    variable = getter(*args, **kwargs)
    trainable = kwargs.get("trainable", True)
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  def build(self, input_shape):
    """Create variables of the Cudnn RNN.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, subsequent `__call__()`s will skip
    creating variables.
    Args:
      input_shape: network input tensor shape, a python list or a TensorShape
        object with 3 dimensions.
    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    if self.built:
      return

    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape.ndims != 3:
      raise ValueError("Expecting input_shape with 3 dims, got %d" %
                       input_shape.ndims)
    if input_shape[-1].value is None:
      raise ValueError("The last dimension of the inputs to `CudnnRNN` "
                       "should be defined. Found `None`.")
    self._input_size = input_shape[-1].value
    self.input_spec = base_layer.InputSpec(ndim=3, axes={-1: self._input_size})

    self._set_scope(None)

    # Not using base class `add_variable()` since the it calls
    # `tf.get_variable()` with a callable initializer whereas here with a
    # tensor. The difference is mandated to support forward-compatibility with
    # Cudnn.
    with vs.variable_scope(
        self._scope,
        reuse=self.built,
        custom_getter=self._update_trainable_weights):
      if self._kernel_initializer is None:
        self._kernel_initializer = init_ops.glorot_uniform_initializer(
            seed=self._seed, dtype=self._plain_dtype)
      if self._bias_initializer is None:
        self._bias_initializer = init_ops.constant_initializer(
            0.0, dtype=self._plain_dtype)

      weights = [
          self._kernel_initializer(sp, dtype=self._plain_dtype)
          for sp in self.canonical_weight_shapes
      ]
      biases = [
          self._bias_initializer(sp, dtype=self._plain_dtype)
          for sp in self.canonical_bias_shapes
      ]
      opaque_params_t = self._canonical_to_opaque(weights, biases)

      if vs.get_variable_scope().partitioner is not None:
        logging.warn(
            "Partitioner is not supported for Cudnn RNN layer variables, using "
            "it will create forward-compatibility issues with future "
            "CUDA/CuDNN generations.")
      # Initialize opaque params with a tensor.
      self.kernel = vs.get_variable(
          "opaque_kernel", initializer=opaque_params_t, validate_shape=False)
    # Create saveable in the outer scope of the cudnn subgraph, such that
    # alternative subgraph with platform-independent rnn cells can load the
    # checkpoints directly.
    if not (self.built or vs.get_variable_scope().reuse):
      self._create_saveable()
    self.built = True

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the RNN model.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple of tensor(s) of shape
        `[num_layers * num_dirs, batch_size, num_units]`. If not provided, use
        zero initial states. The tuple size is 2 for LSTM and 1 for other RNNs.
      training: whether this operation will be used in training or inference.
    Returns:
      output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`.
        It is a `concat([fwd_output, bak_output], axis=2)`.
      output_states: a tuple of tensor(s) of the same shape and structure as
        `initial_state`.
    Raises:
      ValueError: initial_state is not a tuple.
    """
    if initial_state is not None and not isinstance(initial_state, tuple):
      raise ValueError("Invalid initial_state type: %s, expecting tuple.",
                       type(initial_state))
    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    batch_size = array_ops.shape(inputs)[1]
    if initial_state is None:
      initial_state = self._zero_state(batch_size)
    if self._rnn_mode == CUDNN_LSTM:
      h, c = initial_state  # pylint:disable=unbalanced-tuple-unpacking,unpacking-non-sequence
    else:
      h, = initial_state  # pylint:disable=unbalanced-tuple-unpacking,unpacking-non-sequence
    h = ops.convert_to_tensor(h, dtype=dtype)
    if self._rnn_mode == CUDNN_LSTM:
      c = ops.convert_to_tensor(c, dtype=dtype)
    else:
      # For model that doesn't take input_c, replace with a dummy tensor.
      c = array_ops.constant([], dtype=dtype)
    outputs, (output_h, output_c) = self._forward(inputs, h, c, self.kernel,
                                                  training)
    if self._rnn_mode == CUDNN_LSTM:
      return outputs, (output_h, output_c)
    else:
      return outputs, (output_h,)

  def state_shape(self, batch_size):
    raise NotImplementedError

  def _zero_state(self, batch_size):
    res = []
    for sp in self.state_shape(batch_size):
      res.append(array_ops.zeros(sp, dtype=self.dtype))
    return tuple(res)

  def _canonical_weight_shape(self, layer):
    """Shapes of Cudnn canonical weight tensors for given layer."""
    if layer < 0 or layer >= self._num_layers:
      raise ValueError("\'layer\' is not valid, got %s, expecting [%d, %d]" %
                       (layer, 0, self._num_layers-1))
    if not self._input_size:
      raise RuntimeError(
          "%s._canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    input_size = self._input_size
    num_units = self._num_units
    num_gates = self._num_params_per_layer // 2
    is_bidi = self._direction == CUDNN_RNN_BIDIRECTION

    if layer == 0:
      wts_applied_on_inputs = [(num_units, input_size)] * num_gates
    else:
      if is_bidi:
        wts_applied_on_inputs = [(num_units, 2 * num_units)] * num_gates
      else:
        wts_applied_on_inputs = [(num_units, num_units)] * num_gates
    wts_applied_on_hidden_states = [(num_units, num_units)] * num_gates
    tf_wts = wts_applied_on_inputs + wts_applied_on_hidden_states
    return tf_wts if not is_bidi else tf_wts * 2

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Cudnn canonical bias tensors for given layer."""
    num_dirs = 1 if self._direction == CUDNN_RNN_UNIDIRECTION else 2
    return [[self._num_units]] * num_dirs * self._num_params_per_layer

  def _canonical_to_opaque(self, cu_weights, cu_biases):
    if not self._input_size:
      raise RuntimeError(
          "%s._canonical_to_opaque invoked before input shape is known" %
          type(self).__name__)
    with ops.device("/gpu:0"):
      return cudnn_rnn_ops.cudnn_rnn_canonical_to_opaque_params(
          rnn_mode=self._rnn_mode,
          num_layers=self._num_layers,
          num_units=self._num_units,
          input_size=self._input_size,
          weights=cu_weights,
          biases=cu_biases,
          input_mode=self._input_mode,
          seed=self._seed,
          dropout=self._dropout,
          direction=self._direction)

  def _forward(self, inputs, h, c, opaque_params, training):
    output, output_h, output_c = cudnn_rnn_ops._cudnn_rnn(  # pylint:disable=protected-access
        inputs,
        h,
        c,
        opaque_params,
        training,
        self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed)
    return output, (output_h, output_c)

  def _create_saveable(self):
    """Create custom saveable for the Cudnn layer.

    Called during layer building process to make sharing checkpoints between
    Cudnn and Cudnn-compatible RNNs easy.
    Returns:
      a `CudnnOpaqueParamsSaveable` object.
    Raises:
      RuntimeError: if any custom saveable is already created for this layer.
    """
    if self._saveable is not None:
      raise RuntimeError("Cudnn saveable already created.")
    self._saveable = self._saveable_cls(  # pylint:disable=not-callable
        opaque_params=self.trainable_variables[0],
        num_layers=self.num_layers,
        num_units=self.num_units,
        input_size=self.input_size,
        input_mode=self.input_mode,
        direction=self.direction,
        scope=vs.get_variable_scope(),
        name="%s_saveable" % self.trainable_variables[0].op.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self._saveable)


class CudnnLSTM(_CudnnRNN):
  """Cudnn implementation of LSTM layer."""
  _rnn_mode = CUDNN_LSTM
  _num_params_per_layer = CUDNN_LSTM_PARAMS_PER_LAYER
  _saveable_cls = cudnn_rnn_ops.CudnnLSTMSaveable

  def state_shape(self, batch_size):
    """Shape of Cudnn LSTM states.

    Shape is a 2-element tuple. Each is
    [num_layers * num_dirs, batch_size, num_units]
    Args:
      batch_size: an int
    Returns:
      a tuple of python arrays.
    """
    return ([self.num_layers * self.num_dirs, batch_size, self.num_units],
            [self.num_layers * self.num_dirs, batch_size, self.num_units])


class _CudnnRNNNoInputC(_CudnnRNN):
  """Abstract simple CudnnRNN layer without input_c."""

  def state_shape(self, batch_size):
    """Shape of the state of Cudnn RNN cells w/o. input_c.

    Shape is a 1-element tuple,
    [num_layers * num_dirs, batch_size, num_units]
    Args:
      batch_size: an int
    Returns:
      a tuple of python arrays.
    """
    return [self.num_layers * self.num_dirs, batch_size, self.num_units],


class CudnnGRU(_CudnnRNNNoInputC):
  """Cudnn implementation of the GRU layer."""
  _rnn_mode = CUDNN_GRU
  _num_params_per_layer = CUDNN_GRU_PARAMS_PER_LAYER
  _saveable_cls = cudnn_rnn_ops.CudnnGRUSaveable


class CudnnRNNTanh(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-tanh layer."""
  _rnn_mode = CUDNN_RNN_TANH
  _num_params_per_layer = CUDNN_RNN_TANH_PARAMS_PER_LAYER
  _saveable_cls = cudnn_rnn_ops.CudnnRNNTanhSaveable


class CudnnRNNRelu(_CudnnRNNNoInputC):
  """Cudnn implementation of the RNN-relu layer."""
  _rnn_mode = CUDNN_RNN_RELU
  _num_params_per_layer = CUDNN_RNN_RELU_PARAMS_PER_LAYER
  _saveable_cls = cudnn_rnn_ops.CudnnRNNReluSaveable
