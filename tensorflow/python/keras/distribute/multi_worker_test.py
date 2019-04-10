# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test multi-worker Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import os
import sys
import threading

from absl.testing import parameterized

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def _mnist_synthetic_dataset(batch_size, steps_per_epoch):
  # train dataset
  x_train = array_ops.ones([batch_size * steps_per_epoch, 28, 28, 1],
                           dtype=dtypes.float32)
  y_train = array_ops.ones([batch_size * steps_per_epoch, 1],
                           dtype=dtypes.int32)
  train_ds = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  train_ds = train_ds.repeat()
  # train_ds = train_ds.shuffle(100)
  train_ds = train_ds.batch(64, drop_remainder=True)

  # eval dataset
  x_test = random_ops.random_uniform([10000, 28, 28, 1], dtype=dtypes.float32)
  y_test = random_ops.random_uniform([10000, 1],
                                     minval=0,
                                     maxval=9,
                                     dtype=dtypes.int32)
  eval_ds = dataset_ops.Dataset.from_tensor_slices((x_test, y_test))
  eval_ds = eval_ds.repeat()
  eval_ds = eval_ds.batch(64, drop_remainder=True)

  return train_ds, eval_ds


def _get_model(input_shape):
  # Define a deterministically-initialized CNN model to recognize MNIST digits,
  # commented out several layers to simplify it.
  model = keras.models.Sequential()
  model.add(
      keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape,
          kernel_initializer=keras.initializers.TruncatedNormal(seed=99)))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Flatten())
  model.add(
      keras.layers.Dense(
          10,
          activation='softmax',
          kernel_initializer=keras.initializers.TruncatedNormal(seed=99)))

  # TODO(yuefengz): optimizer with slot variables doesn't work because of
  # optimizer's bug.
  # TODO(yuefengz): we should not allow non-v2 optimizer.
  model.compile(
      loss=keras.losses.sparse_categorical_crossentropy,
      optimizer=gradient_descent.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


def _clone_and_build_model(model, strategy):
  # The new "original" model in worker 0.
  with strategy.scope():
    cloned_model = models.clone_model(model)

  # Compile and build model.
  if isinstance(model.optimizer, optimizers.TFOptimizer):
    optimizer = model.optimizer
    # TODO(yuefengz): figure out why the optimizer here is still a
    # TFOptimizer.
    while isinstance(optimizer, optimizers.TFOptimizer):
      optimizer = optimizer.optimizer
    optimizer = copy.deepcopy(optimizer)
  else:
    optimizer_config = model.optimizer.get_config()
    optimizer = type(model.optimizer).from_config(optimizer_config)

  cloned_model.compile(
      optimizer,
      model.loss,
      metrics=metrics_module.clone_metrics(model._compile_metrics),
      loss_weights=model.loss_weights,
      sample_weight_mode=model.sample_weight_mode,
      weighted_metrics=metrics_module.clone_metrics(
          model._compile_weighted_metrics))
  return cloned_model


# TODO(b/123918215): Possibly merge this Callback with keras_test.Counter.
class MultiWorkerVerificationCallback(callbacks.Callback):
  """MultiWorkerVerificationCallback verifies the callbacks in multi-worker scheme.

  This Callback is intended to be used for verifying the callback is indeed
  called the correct number of times in various task types.

  Attributes:
    _task_dict: A nested dictionary storing the number of times a callback has
                been called in specific task type, task index, and method name.
                Look up structure is
                task_name -> task_id -> tracking_method_name -> invoke_count
                For example, a _task_dict of
                {
                    'ps': {
                         0: {
                             'on_epoch_begin': 2
                         },
                         1: {
                             'on_epoch_begin': 2
                         }
                    },
                    'worker': {
                         0: {
                             'on_epoch_begin': 2
                         },
                         1: {
                             'on_epoch_begin': 2
                         }
                    }
                }
                indicates the ps task has 'on_epoch_begin' called twice on each
                of the two indices, and likewise for worker task.
  """

  # TODO(rchao): Add other method calls to verify.
  METHODS_TO_VERIFY = ['on_epoch_begin']

  def __init__(self, num_epoch, num_worker):
    """Initialize a MultiWorkerVerificationCallback.

    Args:
      num_epoch: Number of epochs this Callback is expected to be called for.
      num_worker: Number of workers this Callback is expected to be called from.
    """
    super(MultiWorkerVerificationCallback, self).__init__()
    self._num_epoch = num_epoch
    self._num_worker = num_worker
    self._task_dict = {
        key: collections.defaultdict(lambda: collections.defaultdict(int))
        for key in ['ps', 'worker']
    }
    self._lock = threading.Lock()
    self._is_between_graph = None
    self.wrap_methods(self.METHODS_TO_VERIFY)

  @property
  def is_between_graph(self):
    return self._is_between_graph

  @is_between_graph.setter
  def is_between_graph(self, is_between_graph):
    self._is_between_graph = is_between_graph

  def wrap_methods(self, method_names):
    """Wrap methods so that the counts of calls are tracked.

    Args:
      method_names: A list of names of methods to track calls.
    """
    for method_name in method_names:
      method = getattr(self, method_name)

      def wrapped_method(method_to_wrap, name, *arg, **kwargs):
        # Use lock to ensure += operation is thread-safe.
        with self._lock:
          self._task_dict[test_base.get_task_type()][
              test_base.get_task_index()][name] += 1
        method_to_wrap(*arg, **kwargs)

      setattr(self, method_name,
              functools.partial(wrapped_method, method, method_name))

  def verify(self, test_case):
    method_count_dict = {
        method_name: self._num_epoch for method_name in self.METHODS_TO_VERIFY
    }
    assert self._is_between_graph is not None
    if self._is_between_graph:
      # TODO(b/124171024): In between-graph replication, by default only the
      # chief calls callback. Fix this test to cover that, as well as the rare
      # cases where all workers call.
      worker_call_count = {
          i: method_count_dict for i in range(0, self._num_worker)
      }
    else:
      # If in-graph, only the first worker calls callback methods.
      worker_call_count = {0: method_count_dict}
    test_case.assertDictEqual(
        self._task_dict,
        {
            # PS' callback is not supposed to be called.
            'ps': {},
            # Each of the Worker should be called num_epoch of times.
            'worker': worker_call_count
        })


# TODO(yuefengz): right now, fit or evaluate has to be called under distribution
# strategy's scope.
def _run_standalone_client(test_obj, strategy, cluster_spec):
  input_shape = (28, 28, 1)
  with strategy.scope():
    orig_model = _get_model(input_shape)

  def worker_fn(strategy):
    with ops.Graph().as_default():
      batch_size = 64
      steps = 2

      with strategy.scope():
        train_ds, _ = _mnist_synthetic_dataset(batch_size, steps)
        model = _clone_and_build_model(orig_model, strategy)

        orig_loss, orig_acc = model.evaluate(train_ds, steps=steps)

        # Workaround for the metrics issue (b/122928955) in async training. This
        # can only be used in standalone client mode.
        dc_context.get_current_worker_context().wait_for_other_workers()

        model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)

        dc_context.get_current_worker_context().wait_for_other_workers()

        trained_loss, trained_acc = model.evaluate(train_ds, steps=steps)

      test_obj.assertLessEqual(trained_loss, orig_loss)
      test_obj.assertGreaterEqual(trained_acc, orig_acc)

  dc.run_distribute_coordinator(
      worker_fn,
      strategy,
      mode=dc.CoordinatorMode.STANDALONE_CLIENT,
      cluster_spec=cluster_spec)


def get_strategy_object(strategy_cls):
  if strategy_cls == mirrored_strategy.MirroredStrategy:
    return strategy_cls(mirrored_strategy.all_local_devices())
  else:
    # CollectiveAllReduceStrategy and ParameterServerStrategy.
    return strategy_cls()


class KerasMultiWorkerTestStandaloneClient(test.TestCase,
                                           parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    super(KerasMultiWorkerTestStandaloneClient, cls).setUpClass()
    cls._cluster_spec = test_base.create_in_process_cluster(
        num_workers=2, num_ps=1, has_eval=False)

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[
              mirrored_strategy.MirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy,
              collective_strategy.CollectiveAllReduceStrategy,
          ],
          required_gpus=[0, 1]))
  def testSimpleModelStandaloneClient(self, strategy_cls):
    # With standalone client, training_utils.should_run_multi_worker returns
    # False which means the distribute coordinator won't be called again in
    # `fit`. This is still correct and intended since session is still
    # configured under distribute coordinator's worker context and distribution
    # strategy object is already configured by distribute coordinator for
    # multi-worker training.
    # The logic should be much clearer once standalone client is merged into
    # core Keras as well.
    strategy = get_strategy_object(strategy_cls)

    _run_standalone_client(self, strategy, self._cluster_spec)


class KerasMultiWorkerTestIndependentWorker(test_base.IndependentWorkerTestBase,
                                            parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[
              mirrored_strategy.MirroredStrategy,
              collective_strategy.CollectiveAllReduceStrategy,
          ],
          required_gpus=[0, 1]))
  def testSimpleModelIndependentWorkerSync(self, strategy_cls):
    num_workers = 2
    num_epoch = 2

    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    self._barrier = dc._Barrier(2)

    # The verification callback will be shared by multiple threads.
    verification_callback = MultiWorkerVerificationCallback(
        num_epoch=num_epoch, num_worker=num_workers)

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      """Simulates an Independent Worker inside of a thread."""
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        strategy = get_strategy_object(strategy_cls)
        verification_callback.is_between_graph = \
            strategy.extended.experimental_between_graph
        batch_size = 64
        steps = 2
        train_ds, _ = _mnist_synthetic_dataset(batch_size, steps)
        with strategy.scope():
          model = _get_model((28, 28, 1))
        orig_loss, _ = model.evaluate(train_ds, steps=steps)
        callbacks_for_fit = nest.flatten(
            kwargs.get('verification_callback', []))
        history = model.fit(
            x=train_ds,
            epochs=num_epoch,
            steps_per_epoch=steps,
            callbacks=callbacks_for_fit)
        self.assertIsInstance(history, keras.callbacks.History)
        trained_loss, _ = model.evaluate(train_ds, steps=steps)
        self.assertLess(trained_loss, orig_loss)

    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        verification_callback=verification_callback)

    threads_to_join = []
    strategy = get_strategy_object(strategy_cls)
    if strategy.extended.experimental_between_graph:
      for ts in threads.values():
        threads_to_join.extend(ts)
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)
    verification_callback.verify(self)

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[parameter_server_strategy.ParameterServerStrategy],
          required_gpus=[0, 1]))
  def testSimpleModelIndependentWorkerAsync(self, strategy_cls):
    num_workers = 2
    num_epoch = 2
    cluster_spec = test_base.create_cluster_spec(
        num_workers=num_workers, num_ps=2)
    self._barrier = dc._Barrier(4)

    # The verification callback will be shared by multiple threads.
    verification_callback = MultiWorkerVerificationCallback(
        num_epoch=num_epoch, num_worker=num_workers)

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      """Simulates an Independent Worker inside of a thread."""
      # TODO(rchao/yuefengz): The following is run by both worker and ps
      # threads. The distribute coordinator should run std server immediately
      # without configuring the session (or building the graph) on PS.
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        batch_size = 64
        steps = 2
        strategy = strategy_cls()
        verification_callback.is_between_graph = \
            strategy.extended.experimental_between_graph

        train_ds, _ = _mnist_synthetic_dataset(batch_size, steps)
        val_ds, _ = _mnist_synthetic_dataset(batch_size, steps)
        with strategy.scope():
          model = _get_model((28, 28, 1))

          # TODO(b/123868066): Verify callback for model.evaluate().
          callbacks_for_fit = nest.flatten(
              kwargs.get('verification_callback', []))
          history = model.fit(
              x=train_ds,
              epochs=num_epoch,
              steps_per_epoch=steps,
              validation_data=val_ds,
              validation_steps=steps,
              callbacks=callbacks_for_fit)
        self.assertIsInstance(history, keras.callbacks.History)

    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn,
        cluster_spec,
        verification_callback=verification_callback)

    threads_to_join = []
    for task_type, ts in threads.items():
      # This test can finish once the worker threads complete, and thus
      # the ps threads don't need to be joined.
      if task_type == 'ps':
        continue
      threads_to_join.extend(ts)
    self.join_independent_workers(threads_to_join)
    verification_callback.verify(self)


if __name__ == '__main__':
  # Enable manual variable initialization to make sure variables are initialized
  # by `init_restore_or_wait_for_variables`.
  backend.manual_variable_initialization(True)
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
