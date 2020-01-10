# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Callbacks: utilities called at certain points during model training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import profiler
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=['keras.callbacks.TensorBoard'])
class TensorBoard(callbacks.Callback):
  # pylint: disable=line-too-long
  """Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:
  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```sh
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Arguments:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_grads: whether to visualize gradient histograms in TensorBoard.
        `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network for histograms
        computation.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding layers
        will be saved. If set to 0, embeddings won't be computed. Data to be
        visualized in TensorBoard's Embedding tab must be passed as
        `embeddings_data`.
      embeddings_layer_names: a list of names of layers to keep eye on. If None
        or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
      embeddings_data: data to be embedded at layers specified in
        `embeddings_layer_names`. Numpy array (if the model has a single input)
        or list of Numpy arrays (if the model has multiple inputs). Learn [more
        about
            embeddings](https://www.tensorflow.org/programmers_guide/embedding)
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        samples. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.

  @compatibility(eager)
  Using the `TensorBoard` callback will work when eager execution is enabled,
  with the restriction that outputting histogram summaries of weights and
  gradients is not supported. Consequently, `histogram_freq` will be ignored.
  @end_compatibility
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='./logs',
               histogram_freq=0,
               batch_size=32,
               write_graph=True,
               write_grads=False,
               write_images=False,
               embeddings_freq=0,
               embeddings_layer_names=None,
               embeddings_metadata=None,
               embeddings_data=None,
               update_freq='epoch',
               profile_batch=2):
    super(TensorBoard, self).__init__()
    self.log_dir = log_dir
    self.histogram_freq = histogram_freq
    if self.histogram_freq and context.executing_eagerly():
      logging.warning(
          UserWarning('Weight and gradient histograms not supported for eager'
                      'execution, setting `histogram_freq` to `0`.'))
      self.histogram_freq = 0
    self.merged = None
    self.write_graph = write_graph
    self.write_grads = write_grads
    self.write_images = write_images
    self.batch_size = batch_size
    self._current_batch = 0
    self._total_batches_seen = 0
    self._total_val_batches_seen = 0
    self.embeddings_freq = embeddings_freq
    self.embeddings_layer_names = embeddings_layer_names
    self.embeddings_metadata = embeddings_metadata
    self.embeddings_data = embeddings_data
    if update_freq == 'batch':
      self.update_freq = 1
    else:
      self.update_freq = update_freq
    self._samples_seen = 0
    self._samples_seen_at_last_write = 0
    # TODO(fishx): Add a link to the full profiler tutorial.
    self._profile_batch = profile_batch
    # One profiler session is running if it is True.
    self._is_profiling = False

    # TensorBoard should only write summaries on the chief when in a
    # Multi-Worker setting.
    self._chief_worker_only = True

  def _init_writer(self, model):
    """Sets file writer."""
    if context.executing_eagerly():
      self.writer = summary_ops_v2.create_file_writer(self.log_dir)
      if not model.run_eagerly and self.write_graph:
        with self.writer.as_default():
          summary_ops_v2.graph(K.get_graph(), step=0)
    elif self.write_graph:
      self.writer = tf_summary.FileWriter(self.log_dir, K.get_graph())
    else:
      self.writer = tf_summary.FileWriter(self.log_dir)

  def _make_histogram_ops(self, model):
    """Defines histogram ops when histogram_freq > 0."""
    # only make histogram summary op if it hasn't already been made
    if self.histogram_freq and self.merged is None:
      for layer in self.model.layers:
        for weight in layer.weights:
          mapped_weight_name = weight.name.replace(':', '_')
          tf_summary.histogram(mapped_weight_name, weight)
          if self.write_images:
            w_img = array_ops.squeeze(weight)
            shape = K.int_shape(w_img)
            if len(shape) == 2:  # dense layer kernel case
              if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = K.int_shape(w_img)
              w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
            elif len(shape) == 3:  # convnet case
              if K.image_data_format() == 'channels_last':
                # switch to channels_first to display
                # every kernel as a separate image
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
              w_img = array_ops.reshape(w_img,
                                        [shape[0], shape[1], shape[2], 1])
            elif len(shape) == 1:  # bias case
              w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
            else:
              # not possible to handle 3D convnets etc.
              continue

            shape = K.int_shape(w_img)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            tf_summary.image(mapped_weight_name, w_img)

        if self.write_grads:
          for weight in layer.trainable_weights:
            mapped_weight_name = weight.name.replace(':', '_')
            grads = model.optimizer.get_gradients(model.total_loss, weight)

            def is_indexed_slices(grad):
              return type(grad).__name__ == 'IndexedSlices'

            grads = [
                grad.values if is_indexed_slices(grad) else grad
                for grad in grads
            ]
            tf_summary.histogram('{}_grad'.format(mapped_weight_name), grads)

        if hasattr(layer, 'output'):
          if isinstance(layer.output, list):
            for i, output in enumerate(layer.output):
              tf_summary.histogram('{}_out_{}'.format(layer.name, i), output)
          else:
            tf_summary.histogram('{}_out'.format(layer.name), layer.output)

  def set_model(self, model):
    """Sets Keras model and creates summary ops."""

    self.model = model
    self._init_writer(model)
    # histogram summaries only enabled in graph mode
    if not context.executing_eagerly():
      self._make_histogram_ops(model)
      self.merged = tf_summary.merge_all()

    # If both embedding_freq and embeddings_data are available, we will
    # visualize embeddings.
    if self.embeddings_freq and self.embeddings_data is not None:
      # Avoid circular dependency.
      from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
      self.embeddings_data = training_utils.standardize_input_data(
          self.embeddings_data, model.input_names)

      # If embedding_layer_names are not provided, get all of the embedding
      # layers from the model.
      embeddings_layer_names = self.embeddings_layer_names
      if not embeddings_layer_names:
        embeddings_layer_names = [
            layer.name
            for layer in self.model.layers
            if type(layer).__name__ == 'Embedding'
        ]

      self.assign_embeddings = []
      embeddings_vars = {}

      self.batch_id = batch_id = array_ops.placeholder(dtypes.int32)
      self.step = step = array_ops.placeholder(dtypes.int32)

      for layer in self.model.layers:
        if layer.name in embeddings_layer_names:
          embedding_input = self.model.get_layer(layer.name).output
          embedding_size = np.prod(embedding_input.shape[1:])
          embedding_input = array_ops.reshape(embedding_input,
                                              (step, int(embedding_size)))
          shape = (self.embeddings_data[0].shape[0], int(embedding_size))
          embedding = variables.Variable(
              array_ops.zeros(shape), name=layer.name + '_embedding')
          embeddings_vars[layer.name] = embedding
          batch = state_ops.assign(embedding[batch_id:batch_id + step],
                                   embedding_input)
          self.assign_embeddings.append(batch)

      self.saver = saver.Saver(list(embeddings_vars.values()))

      # Create embeddings_metadata dictionary
      if isinstance(self.embeddings_metadata, str):
        embeddings_metadata = {
            layer_name: self.embeddings_metadata
            for layer_name in embeddings_vars.keys()
        }
      else:
        # If embedding_metadata is already a dictionary
        embeddings_metadata = self.embeddings_metadata

      try:
        from tensorboard.plugins import projector
      except ImportError:
        raise ImportError('Failed to import TensorBoard. Please make sure that '
                          'TensorBoard integration is complete."')

      # TODO(psv): Add integration tests to test embedding visualization
      # with TensorBoard callback. We are unable to write a unit test for this
      # because TensorBoard dependency assumes TensorFlow package is installed.
      config = projector.ProjectorConfig()
      for layer_name, tensor in embeddings_vars.items():
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor.name

        if (embeddings_metadata is not None and
            layer_name in embeddings_metadata):
          embedding.metadata_path = embeddings_metadata[layer_name]

      projector.visualize_embeddings(self.writer, config)

  def _fetch_callback(self, summary):
    self.writer.add_summary(summary, self._total_val_batches_seen)
    self._total_val_batches_seen += 1

  def _write_custom_summaries(self, step, logs=None):
    """Writes metrics out as custom scalar summaries.

    Arguments:
        step: the global step to use for TensorBoard.
        logs: dict. Keys are scalar summary names, values are
            NumPy scalars.

    """
    logs = logs or {}
    if context.executing_eagerly():
      # use v2 summary ops
      with self.writer.as_default(), summary_ops_v2.always_record_summaries():
        for name, value in logs.items():
          if isinstance(value, np.ndarray):
            value = value.item()
          summary_ops_v2.scalar(name, value, step=step)
    else:
      # use FileWriter from v1 summary
      for name, value in logs.items():
        if isinstance(value, np.ndarray):
          value = value.item()
        summary = tf_summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        self.writer.add_summary(summary, step)
    self.writer.flush()

  def on_batch_end(self, batch, logs=None):
    """Writes scalar summaries for metrics on every training batch.

    Performs profiling if current batch is in profiler_batches.
    """
    # Don't output batch_size and batch number as TensorBoard summaries
    logs = logs or {}
    self._samples_seen += logs.get('size', 1)
    samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
    if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
      batch_logs = {('batch_' + k): v
                    for k, v in logs.items()
                    if k not in ['batch', 'size', 'num_steps']}
      self._write_custom_summaries(self._total_batches_seen, batch_logs)
      self._samples_seen_at_last_write = self._samples_seen
    self._total_batches_seen += 1
    if self._is_profiling:
      profiler.save(self.log_dir, profiler.stop())
      self._is_profiling = False
    elif (not self._is_profiling and
          self._total_batches_seen == self._profile_batch - 1):
      profiler.start()
      self._is_profiling = True

  def on_train_begin(self, logs=None):
    if self._profile_batch == 1:
      profiler.start()
      self._is_profiling = True

  def on_epoch_begin(self, epoch, logs=None):
    """Add histogram op to Model eval_function callbacks, reset batch count."""

    # check if histogram summary should be run for this epoch
    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._epoch = epoch
      # pylint: disable=protected-access
      # add the histogram summary op if it should run this epoch
      self.model._make_test_function()
      if self.merged not in self.model.test_function.fetches:
        self.model.test_function.fetches.append(self.merged)
        self.model.test_function.fetch_callbacks[
            self.merged] = self._fetch_callback
      # pylint: enable=protected-access

  def on_epoch_end(self, epoch, logs=None):
    """Checks if summary ops should run next epoch, logs scalar summaries."""

    # don't output batch_size and
    # batch number as TensorBoard summaries
    logs = {('epoch_' + k): v
            for k, v in logs.items()
            if k not in ['batch', 'size', 'num_steps']}
    if self.update_freq == 'epoch':
      step = epoch
    else:
      step = self._samples_seen
    self._write_custom_summaries(step, logs)

    # pop the histogram summary op after each epoch
    if self.histogram_freq:
      # pylint: disable=protected-access
      if self.merged in self.model.test_function.fetches:
        self.model.test_function.fetches.remove(self.merged)
      if self.merged in self.model.test_function.fetch_callbacks:
        self.model.test_function.fetch_callbacks.pop(self.merged)
      # pylint: enable=protected-access

    if self.embeddings_data is None and self.embeddings_freq:
      raise ValueError('To visualize embeddings, embeddings_data must '
                       'be provided.')

    if self.embeddings_freq and self.embeddings_data is not None:
      if epoch % self.embeddings_freq == 0:
        # We need a second forward-pass here because we're passing
        # the `embeddings_data` explicitly. This design allows to pass
        # arbitrary data as `embeddings_data` and results from the fact
        # that we need to know the size of the `tf.Variable`s which
        # hold the embeddings in `set_model`. At this point, however,
        # the `validation_data` is not yet set.

        embeddings_data = self.embeddings_data
        n_samples = embeddings_data[0].shape[0]
        i = 0
        sess = K.get_session()
        while i < n_samples:
          step = min(self.batch_size, n_samples - i)
          batch = slice(i, i + step)

          if isinstance(self.model.input, list):
            feed_dict = {
                model_input: embeddings_data[idx][batch]
                for idx, model_input in enumerate(self.model.input)
            }
          else:
            feed_dict = {self.model.input: embeddings_data[0][batch]}

          feed_dict.update({self.batch_id: i, self.step: step})

          if not isinstance(K.learning_phase(), int):
            feed_dict[K.learning_phase()] = False

          sess.run(self.assign_embeddings, feed_dict=feed_dict)
          self.saver.save(sess,
                          os.path.join(self.log_dir, 'keras_embedding.ckpt'),
                          epoch)

          i += self.batch_size

  def on_train_end(self, logs=None):
    if self._is_profiling:
      profiler.save(self.log_dir, profiler.stop())
      self._is_profiling = False
    self.writer.close()
