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
"""Tests Keras multi worker callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl.testing import parameterized

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import keras
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import multi_worker_test
from tensorflow.python.platform import test

get_strategy_object = multi_worker_test.get_strategy_object
_mnist_synthetic_dataset = multi_worker_test._mnist_synthetic_dataset
_get_model = multi_worker_test._get_model


def generate_callback_test_function(custom_callable):
  """Generic template for callback tests using mnist synthetic dataset."""

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[collective_strategy.CollectiveAllReduceStrategy],
          required_gpus=[0, 1]))
  def test_template(self, strategy_cls):
    num_workers = 2
    num_epoch = 2

    cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
    self._barrier = dc._Barrier(2)

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      """Simulates an Independent Worker inside of a thread."""
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):
        strategy = get_strategy_object(strategy_cls)
        batch_size = 64
        steps = 2
        train_ds, _ = _mnist_synthetic_dataset(batch_size, steps)
        with strategy.scope():
          model = _get_model((28, 28, 1))

        custom_callable(
            model,
            self,
            train_ds,
            num_epoch,
            steps,
            strategy,
            saving_filepath=kwargs['saving_filepath'])

    # Pass saving_filepath from the parent thread to ensure every worker has the
    # same fileapth to save.
    saving_filepath = os.path.join(self.get_temp_dir(), 'checkpoint.h5')
    threads = self.run_multiple_tasks_in_threads(
        _independent_worker_fn, cluster_spec, saving_filepath=saving_filepath)
    if os.path.exists(saving_filepath):
      os.remove(saving_filepath)

    threads_to_join = []
    strategy = get_strategy_object(strategy_cls)
    if strategy.extended.experimental_between_graph:
      for ts in threads.values():
        threads_to_join.extend(ts)
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)

  return test_template


class KerasMultiWorkerCallbackTest(test_base.IndependentWorkerTestBase,
                                   parameterized.TestCase):

  # The callables of the actual testing content to be run go below.
  @staticmethod
  def callableForTestChiefOnlyCallback(model, test_obj, train_ds, num_epoch,
                                       steps, strategy, saving_filepath):

    class ChiefOnly(keras.callbacks.Callback):

      def __init__(self):
        self._chief_worker_only = True
        self.filtered_correctly = True

      def on_train_begin(self, logs):
        if not dc_context.get_current_worker_context().is_chief:
          # Non-chief workers shouldn't run this callback.
          self.filtered_correctly = False

    cb = ChiefOnly()
    model.fit(
        x=train_ds, epochs=num_epoch, steps_per_epoch=steps, callbacks=[cb])

    test_obj.assertTrue(cb.filtered_correctly)

  @staticmethod
  def callableForTestModelCheckpointSavesOnChiefButNotOtherwise(
      model, test_obj, train_ds, num_epoch, steps, strategy, saving_filepath):
    # Incorporate type/index information and thread id in saving_filepath to
    # ensure every worker has a unique path. Note that in normal use case the
    # saving_filepath will be the same for all workers, but we use different
    # ones here just to test out chief saves checkpoint but non-chief doesn't.
    saving_filepath = os.path.join(
        test_obj.get_temp_dir(), 'checkpoint_%s_%d' %
        (test_base.get_task_type(), test_base.get_task_index()))

    # The saving_filepath shouldn't exist at the beginning (as it's unique).
    test_obj.assertFalse(os.path.exists(saving_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[callbacks.ModelCheckpoint(filepath=saving_filepath)])

    # If it's chief, the model should be saved; if not, the model shouldn't.
    test_obj.assertEqual(os.path.exists(saving_filepath), test_base.is_chief())

  @staticmethod
  def initialFitting(test_obj, model, train_ds, num_epoch, steps,
                     saving_filepath):
    # The saving_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(os.path.exists(saving_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath, save_weights_only=True)
        ])

    # The saving_filepath should exist after fitting with callback. Both chief
    # and non-chief worker should both see it exists (which was saved only by
    # chief).
    test_obj.assertTrue(os.path.exists(saving_filepath))

    history_after_one_more_epoch = model.fit(
        x=train_ds, epochs=1, steps_per_epoch=steps)

    # The saving_filepath should continue to exist (if it did) after fitting
    # without callback.
    test_obj.assertTrue(os.path.exists(saving_filepath))

    return saving_filepath, history_after_one_more_epoch

  @staticmethod
  def callableForTestLoadWeightFromModelCheckpoint(model, test_obj, train_ds,
                                                   num_epoch, steps, strategy,
                                                   saving_filepath):

    saving_filepath, history_after_one_more_epoch = \
        KerasMultiWorkerCallbackTest.initialFitting(
            test_obj, model, train_ds, num_epoch, steps, saving_filepath)

    with strategy.scope():
      model.load_weights(saving_filepath)

    history_after_loading_weight_and_one_more_epoch = model.fit(
        x=train_ds, epochs=1, steps_per_epoch=steps)

    test_obj.assertAllClose(
        history_after_one_more_epoch.history,
        history_after_loading_weight_and_one_more_epoch.history)

  @staticmethod
  def callableForTestModelRestoreCallback(model, test_obj, train_ds, num_epoch,
                                          steps, strategy, saving_filepath):

    saving_filepath, history_after_one_more_epoch = \
        KerasMultiWorkerCallbackTest.initialFitting(
            test_obj, model, train_ds, num_epoch, steps, saving_filepath)

    # The model should get restored to the weights previously saved, by
    # adding a ModelCheckpoint callback (which results in a
    # _ModelRestoreCallback being added), with load_weights_on_restart=True.
    history_after_model_restoring_and_one_more_epoch = model.fit(
        x=train_ds,
        epochs=1,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath,
                save_weights_only=True,
                load_weights_on_restart=True)
        ])

    # Asserting the history one epoch after initial fitting and one epoch after
    # restoring are closed.
    test_obj.assertAllClose(
        history_after_one_more_epoch.history,
        history_after_model_restoring_and_one_more_epoch.history)

  @staticmethod
  def callableForTestUnmatchedModelFile(model, test_obj, train_ds, num_epoch,
                                        steps, strategy, saving_filepath):

    # The saving_filepath shouldn't exist at the beginning.
    test_obj.assertFalse(os.path.exists(saving_filepath))

    model.fit(
        x=train_ds,
        epochs=num_epoch,
        steps_per_epoch=steps,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=saving_filepath, save_weights_only=True)
        ])

    (train_ds, _), (_, _) = testing_utils.get_test_data(
        train_samples=10, test_samples=10, input_shape=(3,), num_classes=2)

    # Switch to a model of different structure.
    with strategy.scope():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(5, input_dim=3, activation='relu'))
      model.add(keras.layers.Dense(2, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    # TODO(b/129779608): Fix the flakiness of the following check.
    # test_obj.assertTrue(os.path.exists(saving_filepath))

    # Unmatched format. Should raise ValueError.
    with test_obj.assertRaisesRegexp(ValueError, 'Error loading file from'):
      model.fit(
          x=train_ds,
          epochs=num_epoch,
          batch_size=8,
          callbacks=[
              callbacks.ModelCheckpoint(
                  filepath=saving_filepath,
                  save_weights_only=True,
                  load_weights_on_restart=True)
          ])

  # The actual testing methods go here.
  test_chief_only_callback = generate_callback_test_function(
      callableForTestChiefOnlyCallback.__func__)
  test_model_checkpoint_saves_on_chief_but_not_otherwise = \
      generate_callback_test_function(
          callableForTestModelCheckpointSavesOnChiefButNotOtherwise.__func__)
  test_load_weight_from_model_checkpoint = generate_callback_test_function(
      callableForTestLoadWeightFromModelCheckpoint.__func__)
  test_model_restore_callback = generate_callback_test_function(
      callableForTestModelRestoreCallback.__func__)
  test_unmatched_model_file = generate_callback_test_function(
      callableForTestUnmatchedModelFile.__func__)


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
