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
"""*Experimental* support for running Keras models on the TPU.

To use, wrap your model with the `keras_support.tpu_model` function.

Example usage:

```
# Must activate before building TPU models
keras_support.setup_tpu_session(master_address)

image = tf.keras.layers.Input(shape=(28, 28, 3), name='image')
c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))( image)
flattened = tf.keras.layers.Flatten()(c1)
logits = tf.keras.layers.Dense(10, activation='softmax')(flattened)
model = tf.keras.Model(inputs=[image], outputs=[logits])
model = keras_support.tpu_model(model)

# Only TF optimizers are currently supported.
model.compile(optimizer=tf.train.AdamOptimizer(), ...)

# `images` and `labels` should be Numpy arrays.  Support for tensor input
# (e.g. datasets) is planned.
model.fit(images, labels)

# Invoke before shutting down
keras_support.shutdown_tpu_session()
```
"""

# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers
from tensorflow.python.keras._impl.keras import models
from tensorflow.python.keras._impl.keras import optimizers as keras_optimizers
from tensorflow.python.keras._impl.keras.layers import embeddings
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util


class TPUEmbedding(embeddings.Embedding):
  """TPU compatible embedding layer.

  The default Keras layer is not TPU compatible.  This layer is a drop-in
  replacement: it has the same behavior and will work on CPU and GPU devices.
  """

  def __init__(self, *args, **kw):
    super(TPUEmbedding, self).__init__(*args, **kw)

  def build(self, input_shape):
    if input_shape[0] is None:
      raise ValueError(
          'TPUEmbeddings must have a fixed input_length or input shape.')
    return super(TPUEmbedding, self).build(input_shape)

  def call(self, inputs):
    if K.dtype(inputs) != 'int32':
      inputs = math_ops.cast(inputs, 'int32')

    inputs = array_ops.one_hot(inputs, self.input_dim)
    return math_ops.tensordot(inputs, self.embeddings, 1)


class CompiledTPUOp(
    collections.namedtuple(
        'CompiledTPUOp',
        ['tpu_execute_op', 'infeed_tensors', 'infeed_op', 'outfeed_op'])):
  pass


def _valid_name(tensor_name):
  """Return a valid tensor name (strips '/', ':', etc)."""
  return re.sub('[^a-zA-Z0-9_-]+', '', tensor_name)


class TPUFunction(object):
  """K.function compatible interface for invoking a TPU compiled function.

  Recompilation is triggered on-demand for each set of new inputs shapes: the
  results are cached for future execution.  We expect most computations will
  be dominated by a standard batch-size, followed by a straggler batch for
  the end of training or evaluation.

  All `inputs` and `outputs` will be loaded via the infeed and outfeed queues
  instead of being injected as `feed_dict` items or fetches.
  """

  def __init__(self, model, execution_mode):
    self.model = model
    self.execution_mode = execution_mode
    self._compilation_cache = {}

  def _specialize_model(self, input_specs):
    """Specialize `self.model` (a Keras model) for the given input shapes."""
    # Re-create our input and output layers inside our subgraph.  They will be
    # attached to the true computation when we clone our model in `tpu_fn`.
    K.set_learning_phase(self.execution_mode == model_fn_lib.ModeKeys.TRAIN)

    # functools.partial and callable objects are not supported by tpu.rewrite
    def _model_fn():
      """Compute fit/eval/predict for the TPU."""
      is_training = self.execution_mode == model_fn_lib.ModeKeys.TRAIN
      is_test = self.execution_mode == model_fn_lib.ModeKeys.EVAL
      is_predict = self.execution_mode == model_fn_lib.ModeKeys.PREDICT

      # During train/eval, we infeed our features as well as labels.
      if is_training or is_test:
        infeed_layers = self.model._input_layers + self.model._output_layers
      else:
        infeed_layers = self.model._input_layers

      # Generate our infeed operation to read features & labels.
      infeed_tensors = tpu_ops.infeed_dequeue_tuple(
          dtypes=[spec.dtype for spec in input_specs],
          shapes=[spec.shape for spec in input_specs],
          name='infeed-%s' % self.execution_mode)

      assert len(infeed_tensors) == len(infeed_layers), (
          'Infeed inputs did not match model: %s vs %s', (infeed_layers,
                                                          infeed_tensors))

      tpu_targets = []
      tpu_inputs = []

      # Sort infeed outputs into inputs and labels for calling our Keras model.
      for tensor, layer in zip(infeed_tensors, infeed_layers):
        if layer in self.model._input_layers:
          tpu_inputs.append(layers.Input(name=layer.name, tensor=tensor))
        if layer in self.model._output_layers:
          tpu_targets.append(tensor)

      optimizer = self.model.optimizer
      optimizer.iterations = training_util.get_or_create_global_step()

      # Call our model with our infeed inputs (re-using the weights).
      model_outputs = self.model(tpu_inputs)
      child_model = models.Model(inputs=tpu_inputs, outputs=model_outputs)
      if is_training or is_test:
        child_model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            loss_weights=self.model.loss_weights,
            metrics=self.model.metrics,
            weighted_metrics=self.model.weighted_metrics,
            target_tensors=tpu_targets,
        )

      # Compute our outfeed depending on the execution mode
      if is_training:
        child_model._make_train_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in child_model.train_function.outputs
        ]
        return [
            child_model.train_function.updates_op,
            tpu_ops.outfeed_enqueue_tuple(
                child_model.train_function.outputs, name='oufeed-enqueue-train')
        ]
      elif is_test:
        child_model._make_test_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in child_model.test_function.outputs
        ]
        return [
            tpu_ops.outfeed_enqueue_tuple(
                child_model.test_function.outputs, name='outfeed-enqueue-test')
        ]
      elif is_predict:
        child_model._make_predict_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in child_model.predict_function.outputs
        ]
        return [
            tpu_ops.outfeed_enqueue_tuple(
                child_model.predict_function.outputs,
                name='outfeed-enqueue-predict',
            )
        ]
      else:
        assert False, 'Unexpected execution mode: %s' % self.execution_mode

    # Capture outfeed metadata computed during the rewrite.
    self._outfeed_spec = None

    tpu_execute_op = tpu.rewrite(_model_fn)

    K._initialize_variables(K.get_session())  # pylint-disable: protected-access

    # Generate CPU side operations to enqueue features/labels and dequeue
    # outputs from the model call.
    with ops.device('/device:TPU:0'):
      infeed_tensors = []
      for spec in input_specs:
        infeed_tensors.append(
            array_ops.placeholder(
                dtype=spec.dtype,
                shape=spec.shape,
                name='infeed-enqueue-%s' % spec.name))

      infeed_op = tpu_ops.infeed_enqueue_tuple(
          infeed_tensors, [spec.shape for spec in input_specs],
          name='infeed-enqueue-%s' % self.execution_mode)

      outfeed_op = tpu_ops.outfeed_dequeue_tuple(
          dtypes=[spec.dtype for spec in self._outfeed_spec],
          shapes=[spec.shape for spec in self._outfeed_spec],
          name='outfeed-dequeue-%s' % self.execution_mode)

    return CompiledTPUOp(tpu_execute_op, infeed_tensors, infeed_op, outfeed_op)

  def __call__(self, inputs):
    assert isinstance(inputs, list)

    # Strip sample weight from inputs
    if (self.execution_mode == model_fn_lib.ModeKeys.TRAIN or
        self.execution_mode == model_fn_lib.ModeKeys.EVAL):
      input_tensors = self.model._feed_inputs + self.model._feed_targets
      inputs = inputs[:len(input_tensors)]
    else:
      input_tensors = self.model._feed_inputs

    # Compute an input specification (used to generate infeed enqueue and
    # dequeue operations).  We use the shape from our input array and the
    # dtype from our model.  A user may pass in a float64 for a float32
    # input: for model compatibility we still must generate a float32 infeed.
    input_specs = []
    for tensor, ary in zip(input_tensors, inputs):
      input_specs.append(
          tensor_spec.TensorSpec(ary.shape, tensor.dtype,
                                 _valid_name(tensor.name)))

    # XLA requires every operation in the graph has a fixed shape.  To
    # handle varying batch sizes we recompile a new sub-graph for each
    # unique input shape.
    shape_key = tuple([tuple(spec.shape.as_list()) for spec in input_specs])

    if shape_key not in self._compilation_cache:
      logging.info('New input shapes; (re-)compiling: mode=%s, %s',
                   self.execution_mode, input_specs)
      self._compilation_cache[shape_key] = self._specialize_model(input_specs)

    compiled_model = self._compilation_cache[shape_key]

    infeed_dict = {}
    for tensor, value in zip(compiled_model.infeed_tensors, inputs):
      infeed_dict[tensor] = value

    session = K.get_session()
    _, _, outfeed_outputs = session.run([
        compiled_model.infeed_op, compiled_model.tpu_execute_op,
        compiled_model.outfeed_op
    ], infeed_dict)

    return outfeed_outputs


@experimental
def setup_tpu_session(master):
  """Initializes and returns a Keras/TF session connected the TPU `master`."""
  session = tf_session.Session(
      target=master, config=config_pb2.ConfigProto(isolate_session_state=True))
  K.set_session(session)
  K.get_session().run(tpu.initialize_system())
  K.manual_variable_initialization(True)
  return session


@experimental
def shutdown_tpu_session(session=None):
  """Shutdown the TPU attached to session.

  This should be called to cleanly shut down the TPU system before the client
  exits.

  Args:
    session: Session to shutdown, or None to use the default session.

  Returns:

  """
  if session is None:
    session = K.get_session()

  session.run(tpu.shutdown_system())


class KerasTPUModel(models.Model):
  """TPU compatible Keras model wrapper."""

  def __init__(self, inputs, outputs, name=None):
    super(models.Model, self).__init__(
        inputs=inputs,
        outputs=outputs,
        name=name,
    )
    self.predict_function = None
    self.test_function = None
    self.train_function = None

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    if sample_weight_mode:
      raise ValueError('sample_weight_mode not supported for TPU execution.')
    if weighted_metrics:
      raise ValueError('weighted_metrics not supported for TPU execution.')
    if target_tensors:
      raise ValueError('target_tensors is not supported for TPU execution.')

    super(KerasTPUModel, self).compile(optimizer, loss, metrics, loss_weights,
                                       sample_weight_mode, weighted_metrics,
                                       target_tensors, **kwargs)

    # Keras optimizers are not compatible with TPU rewrite
    if not isinstance(self.optimizer, keras_optimizers.TFOptimizer):
      raise ValueError(
          'Optimizer must be a TFOptimizer, got: %s' % self.optimizer)

  def train_on_batch(self, x, y, sample_weight=None, class_weight=None):
    return super(KerasTPUModel, self).train_on_batch(x, y, sample_weight,
                                                     class_weight)

  def _make_train_function(self):
    if not self.train_function:
      self.train_function = TPUFunction(self, model_fn_lib.ModeKeys.TRAIN)

    return self.train_function

  def _make_test_function(self):
    if not self.test_function:
      self.test_function = TPUFunction(self, model_fn_lib.ModeKeys.EVAL)
    return self.test_function

  def _make_predict_function(self):
    if not self.predict_function:
      self.predict_function = TPUFunction(self, model_fn_lib.ModeKeys.PREDICT)
    return self.predict_function

  def cpu_model(self):
    return models.Model(
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
    )


@experimental
def tpu_model(model):
  return KerasTPUModel(
      inputs=model.inputs, outputs=model.outputs, name=model.name)
