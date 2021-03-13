# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test for multi-worker training tutorial."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import os
import re
import unittest
import uuid
import zipfile
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

PER_WORKER_BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_EPOCHS = 2
NUM_STEPS_PER_EPOCH = 50


def _is_chief(task_type, task_id):
  # Note: there are two possible `TF_CONFIG` configuration.
  #   1) In addition to `worker` tasks, a `chief` task type is use;
  #      in this case, this function should be modified to
  #      `return task_type == 'chief'`.
  #   2) Only `worker` task type is used; in this case, worker 0 is
  #      regarded as the chief. The implementation demonstrated here
  #      is for this case.
  return task_type == 'worker' and task_id == 0


def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir


def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)


class MultiWorkerTutorialTest(parameterized.TestCase, tf.test.TestCase):
  """Test of multi-worker training flow in tutorials on tensorflow.org.

  Please see below test method docs for what actual tutorial is being covered.
  """

  # TODO(rchao): Add a test to demonstrate gather with MWMS.

  @contextlib.contextmanager
  def skip_fetch_failure_exception(self):
    try:
      yield
    except zipfile.BadZipfile as e:
      # There can be a race when multiple processes are downloading the data.
      # Skip the test if that results in loading errors.
      self.skipTest('Data loading error: Bad magic number for file header.')
    except Exception as e:  # pylint: disable=broad-except
      if 'URL fetch failure' in str(e):
        self.skipTest('URL fetch error not considered failure of the test.')
      else:
        raise

  def mnist_dataset(self):
    path_to_use = 'mnist_{}.npz'.format(str(uuid.uuid4()))
    with self.skip_fetch_failure_exception():
      (x_train,
       y_train), _ = tf.keras.datasets.mnist.load_data(path=path_to_use)
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000)
    return train_dataset

  def dataset_fn(self, global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = self.mnist_dataset()
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    return dataset

  def build_cnn_model(self):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

  def build_and_compile_cnn_model(self):
    model = self.build_cnn_model()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

  @tf.__internal__.test.combinations.generate(
      tf.__internal__.test.combinations.combine(
          mode=['eager'], tf_api_version=2))
  def testSingleWorkerModelFit(self):
    single_worker_dataset = self.mnist_dataset().batch(
        PER_WORKER_BATCH_SIZE)
    single_worker_model = self.build_and_compile_cnn_model()
    single_worker_model.fit(single_worker_dataset, epochs=NUM_EPOCHS)

  @tf.__internal__.test.combinations.generate(
      tf.__internal__.test.combinations.combine(
          mode=['eager'], tf_api_version=2))
  def testMwmsWithModelFit(self, mode):
    """Test multi-worker training flow demo'ed in go/multi-worker-with-keras.

    This test should be kept in sync with the code samples in
    go/multi-worker-with-keras.

    Args:
      mode: Runtime mode.
    """
    def fn(model_path, checkpoint_dir):
      global_batch_size = PER_WORKER_BATCH_SIZE * NUM_WORKERS
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
      with strategy.scope():
        multi_worker_model = self.build_and_compile_cnn_model()

      callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
              filepath=os.path.join(self.get_temp_dir(), 'checkpoint'))
      ]

      multi_worker_dataset = strategy.distribute_datasets_from_function(
          lambda input_context: self.dataset_fn(global_batch_size, input_context
                                               ))

      multi_worker_model.fit(
          multi_worker_dataset,
          epochs=NUM_EPOCHS,
          steps_per_epoch=50,
          callbacks=callbacks)

      task_type, task_id = (strategy.cluster_resolver.task_type,
                            strategy.cluster_resolver.task_id)
      write_model_path = write_filepath(model_path, task_type, task_id)

      multi_worker_model.save(write_model_path)
      if not _is_chief(task_type, task_id):
        tf.io.gfile.rmtree(os.path.dirname(write_model_path))

      # Make sure chief finishes saving before non-chief's assertions.
      tf.__internal__.distribute.multi_process_runner.get_barrier().wait()

      if not tf.io.gfile.exists(model_path):
        raise RuntimeError()
      if tf.io.gfile.exists(write_model_path) != _is_chief(task_type, task_id):
        raise RuntimeError()

      with strategy.scope():
        loaded_model = tf.keras.models.load_model(model_path)
      loaded_model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=1)

      checkpoint = tf.train.Checkpoint(model=multi_worker_model)
      write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id)
      checkpoint_manager = tf.train.CheckpointManager(
          checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

      checkpoint_manager.save()
      if not _is_chief(task_type, task_id):
        tf.io.gfile.rmtree(write_checkpoint_dir)

      # Make sure chief finishes saving before non-chief's assertions.
      tf.__internal__.distribute.multi_process_runner.get_barrier().wait()

      if not tf.io.gfile.exists(checkpoint_dir):
        raise RuntimeError()
      if tf.io.gfile.exists(write_checkpoint_dir) != _is_chief(
          task_type, task_id):
        raise RuntimeError()

      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
      checkpoint.restore(latest_checkpoint)
      multi_worker_model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=1)

      logging.info('testMwmsWithModelFit successfully ends')

    model_path = os.path.join(self.get_temp_dir(), 'model.tf')
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'ckpt')
    try:
      mpr_result = tf.__internal__.distribute.multi_process_runner.run(
          fn,
          tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
              num_workers=NUM_WORKERS),
          args=(model_path, checkpoint_dir),
          return_output=True)
    except tf.errors.UnavailableError:
      self.skipTest('Skipping rare disconnection among the workers.')

    self.assertTrue(
        any([
            'testMwmsWithModelFit successfully ends' in msg
            for msg in mpr_result.stdout
        ]))

    def extract_accuracy(worker_id, input_string):
      match = re.match(
          r'\[worker\-{}\].*accuracy: (\d+\.\d+).*'.format(worker_id),
          input_string)
      return None if match is None else float(match.group(1))

    for worker_id in range(NUM_WORKERS):
      accu_result = tf.nest.map_structure(
          lambda x: extract_accuracy(worker_id, x),  # pylint: disable=cell-var-from-loop
          mpr_result.stdout)
      self.assertTrue(
          any(accu_result), 'Every worker is supposed to have accuracy result.')

  @tf.__internal__.test.combinations.generate(
      tf.__internal__.test.combinations.combine(
          mode=['eager'], tf_api_version=2))
  def testMwmsWithCtl(self, mode):
    """Test multi-worker CTL training flow demo'ed in a to-be-added tutorial."""

    def proc_func(checkpoint_dir):
      global_batch_size = PER_WORKER_BATCH_SIZE * NUM_WORKERS
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
      try:

        with strategy.scope():
          multi_worker_model = self.build_cnn_model()

        multi_worker_dataset = strategy.distribute_datasets_from_function(
            lambda input_context: self.dataset_fn(global_batch_size,  # pylint: disable=g-long-lambda
                                                  input_context))
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        @tf.function
        def train_step(iterator):
          """Training step function."""

          def step_fn(inputs):
            """Per-Replica step function."""
            x, y = inputs
            with tf.GradientTape() as tape:
              predictions = multi_worker_model(x, training=True)
              per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True,
                  reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
              loss = tf.nn.compute_average_loss(
                  per_batch_loss, global_batch_size=global_batch_size)

            grads = tape.gradient(loss, multi_worker_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, multi_worker_model.trainable_variables))
            train_accuracy.update_state(y, predictions)

            return loss

          per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
          return strategy.reduce(
              tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        epoch = tf.Variable(
            initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
        step_in_epoch = tf.Variable(
            initial_value=tf.constant(0, dtype=tf.dtypes.int64),
            name='step_in_epoch')

        task_type, task_id = (strategy.cluster_resolver.task_type,
                              strategy.cluster_resolver.task_id)
        checkpoint = tf.train.Checkpoint(
            model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)
        write_checkpoint_dir = write_filepath(checkpoint_dir, task_type,
                                              task_id)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
          checkpoint.restore(latest_checkpoint)

        while epoch.numpy() < NUM_EPOCHS:
          iterator = iter(multi_worker_dataset)
          total_loss = 0.0
          num_batches = 0

          while step_in_epoch.numpy() < NUM_STEPS_PER_EPOCH:
            total_loss += train_step(iterator)
            num_batches += 1
            step_in_epoch.assign_add(1)

          train_loss = total_loss / num_batches
          logging.info('Epoch: %d, accuracy: %f, train_loss: %f.',
                       epoch.numpy(), train_accuracy.result(), train_loss)

          train_accuracy.reset_states()

          checkpoint_manager.save()
          if not _is_chief(task_type, task_id):
            tf.io.gfile.rmtree(write_checkpoint_dir)

          epoch.assign_add(1)
          step_in_epoch.assign(0)

      except tf.errors.UnavailableError as e:
        logging.info('UnavailableError occurred: %r', e)
        raise unittest.SkipTest('Skipping test due to UnavailableError')

      logging.info('testMwmsWithCtl successfully ends')

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'ckpt')

    mpr_result = tf.__internal__.distribute.multi_process_runner.run(
        proc_func,
        tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
            num_workers=NUM_WORKERS),
        return_output=True,
        args=(checkpoint_dir,))

    self.assertTrue(
        any([
            'testMwmsWithCtl successfully ends' in msg
            for msg in mpr_result.stdout
        ]))


if __name__ == '__main__':
  tf.__internal__.distribute.multi_process_runner.test_main()
