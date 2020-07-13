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
"""This file contains unit tests of past, existing and potential saving issues with tf.distribute.

The tests are written as minimum reproductions in the hope to demonstrate the
expected behavior in a straightforward fashion.

Test cases ending with _broken are known issues. Assertions in such tests
describes the current incorrect behavior. If you fix something, you should
expect some test cases to fail and please update them.

This file is not intended to provide exhaustive test coverage. Exhaustive tests
using Keras models are in keras*_test.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.tpu_strategy,
        ],
        mode="eager",
    ))
class SaveAndLoadForServingTest(test.TestCase, parameterized.TestCase):
  # These test cases cover the case when a model is trained under
  # tf.distribute.Strategy and used for serving later. Serving usually only uses
  # one device and this is simulated by loading the model under no strategy
  # using tf.saved_model.load. Note that tf.keras.models.load_model doesn't use
  # the saved functions so they're not fit for this test.
  #
  # When saving, it's expected that the model is saved as if there's no
  # tf.distribute.Strategy.  The saved tf.function should be an inference
  # function on a single device, and the distributed variables are saved as
  # single variables.
  #
  # Curently references to components of a distributed variable are mapped to
  # the single variable that is saved. This means that if the saved tf.functions
  # access components of a distributed variable, for example if it triggers
  # variable aggregation, the outputs are likely incorrect.
  #
  # Note that distributed variables have different behavior in the replica
  # context and the cross-replica context. Saving happens in the cross replica
  # context or the default startegy's replica context.

  def test_read_sync_on_read_variable_broken(self, strategy):
    # synchronizaiton=ON_READ variables are typically used in Keras metrics and
    # batch norm layers.
    #
    # This is broken now since the saved variable already has the aggregated
    # value, but the saved tf.function is traced under the cross-replica context
    # and contains the aggregation.
    #
    # Impacts:
    #   - MirroredStrategy, TPUStrategy
    #     - aggregation=NONE: error when saving.
    #     - aggregation=SUM: incorrect results.
    #     - aggregation=MEAN: slight computation overhead.
    #     - aggregation=ONLY_FIRST_REPLICA: none.
    #   - MultiWorkerMirroredStrategy:
    #     - aggregation=NONE: error when saving
    #     - aggregation=MEAN, SUM: error or hanging when using the loaded model.
    #     - aggregation=ONLY_FIRST_REPLICA: none.
    # Note that batch norm uses aggregation=MEAN.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0.,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.SUM)

      @tf.function(input_signature=[])
      def __call__(self):
        return self.v.read_value()

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      # Note that each component is assigned with the value divided by the
      # number of replicas.
      m.v.assign(1.)
      self.assertAllEqual(
          self.evaluate(strategy.experimental_local_results(m.v)), [0.5, 0.5])
      tf.saved_model.save(m, export_dir)

    loaded = tf.saved_model.load(export_dir)
    # The variable already has the aggregated value.
    self.assertEqual(self.evaluate(loaded.v.read_value()), 1.)
    # TODO(b/159752793): reading the variable aggregates the values again.
    # got 2., want 1.
    self.assertEqual(self.evaluate(loaded()), 2.)

  def test_read_mirrored_variable(self, strategy):
    # synchronizaiton=ON_WRITE is the default variable created under
    # tf.distribute.Strategy.scope(). Most model parameters are this kind of
    # variable. Reading a synchronization=ON_WRITE simply reads the primary
    # component, so it works as intended.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0., synchronization=tf.VariableSynchronization.ON_WRITE)

      @tf.function(input_signature=[])
      def __call__(self):
        return self.v.read_value()

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      m.v.assign(1.)
      tf.saved_model.save(m, export_dir)

    loaded = tf.saved_model.load(export_dir)
    self.assertEqual(self.evaluate(loaded()), 1.)

  def test_update_sync_on_read_variable_broken(self, strategy):
    # It's rare to update aggregation=ON_READ variables in serving, but it's
    # possible that the SavedModel contains both serving and training graphs,
    # and the training may contain metrics layers.
    #
    # This is now partially broken since assign_add() and assign_sub() are not
    # allowed in the cross-replica context if aggregation=SUM, which blocks
    # saving the model.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0.,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.SUM)

      @tf.function(input_signature=[])
      def update(self):
        self.v.assign_add(1.)
        return self.v.read_value()

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      # got error, want no error.
      with self.assertRaisesRegex(ValueError,
                                  "SyncOnReadVariable does not support"):
        tf.saved_model.save(m, export_dir)

    # TODO(b/159752793): Uncomment after fix.
    # loaded = tf.saved_model.load(export_dir)
    # loaded.update()
    # self.assertEqual(self.evaluate(loaded.v), 1.)

  def test_update_mirrored_variable_broken(self, strategy):
    # It's very rare to update aggregation=ON_WRITE variables in the forward
    # path, and this test case is mainly for completeness.
    #
    # The saved tf.function updates each components of the distributed variable,
    # which effectively updates the variable in the saved model N times where N
    # equals the number of local replicas during training.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0.,
            synchronization=tf.VariableSynchronization.ON_WRITE,
            aggregation=tf.VariableAggregation.MEAN)

      @tf.function(input_signature=[])
      def update(self):
        self.v.assign_add(1.)

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      tf.saved_model.save(m, export_dir)

    loaded = tf.saved_model.load(export_dir)
    self.assertEqual(self.evaluate(loaded.v), 0.)
    loaded.update()
    # TODO(b/159752793): Change after fix.
    # got 2., want 1.
    self.assertEqual(self.evaluate(loaded.v), 2.)

  def test_training_only_device(self, strategy):
    # tf.distribute APIs may enter device scopes, but the saved model should not
    # contain device annotations, since devices during training may not be
    # available when the saved model is used.
    #
    # Models trained with MultiWorkerMirroredStrategy is affected the most,
    # since under MultiWorkerMirroredStrategy the device annotations contain job
    # and task, which causes error if that job or task is not available even
    # with soft placement enabled.

    class Model(tf.Module):

      @tf.function(input_signature=[])
      def __call__(self):
        return tf.identity(1.)

    export_dir = self.get_temp_dir()
    with strategy.scope(), tf.device("GPU:0"):
      m = Model()
      tf.saved_model.save(m, export_dir)

    export_dir = self.get_temp_dir()
    loaded = tf.saved_model.load(export_dir)
    graph = loaded.signatures["serving_default"].graph
    for op in graph.get_operations():
      self.assertEqual(op.device, "")
    self.assertEqual(loaded().numpy(), 1.)

  def test_model_with_loaded_layer_broken(self, strategy):
    # If a model contains a layer loaded from SavedModel, including tf.hub
    # layers, and if the model is created under tf.distribute.Strategy, it
    # cannot be saved again. The saving won't error but the saved model cannot
    # be used.
    #
    # The reason is that if a saved model is loaded under
    # tf.distribute.Strategy, the tf.functions are wrapped by
    # saved_model._WrapperFunction, which generates an assertion node in the
    # cross-replica context.

    class Layer(tf.Module):

      def __init__(self):
        self.v = tf.Variable(1.)

      @tf.function(input_signature=[])
      def __call__(self):
        return self.v.read_value()

    class Model(tf.Module):

      def __init__(self, layer):
        self.layer = layer

      @tf.function(input_signature=[])
      def __call__(self):
        return self.layer()

    layer_export_dir = self.get_temp_dir()
    tf.saved_model.save(Layer(), layer_export_dir)

    with strategy.scope():
      m = Model(tf.saved_model.load(layer_export_dir))
      export_dir = self.get_temp_dir()
      # It happens to work if we save the model outside of strategy.scope(),
      # because DistributedVariable.handle and _WrapperFunction behaved
      # differently under the cross-replica context and the default strategy's
      # replica context.
      tf.saved_model.save(m, export_dir)

    loaded = tf.saved_model.load(export_dir)
    # got error, want [1., 1.]
    if isinstance(strategy, tf.distribute.MirroredStrategy):
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          "from the cross-replica context in an in-replica context"):
        strategy.run(loaded)
    else:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  "No registered 'Placeholder'"):
        strategy.run(loaded)
    # TODO(b/160646235): Uncomment after fix.
    #  self.assertAllEqual(
    #      self.evaluate(
    #          strategy.experimental_local_results(strategy.run(loaded)),
    #          [1., 1.]))


@combinations.generate(
    combinations.combine(
        strategy=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu],
        mode="eager",
    ))
class SaveAndLoadForTrainingTest(test.TestCase, parameterized.TestCase):
  # These test cases cover the case when the user loads a model and continues to
  # train it. The model could originally be trained with or without
  # tf.distribute.Strategy.
  #
  # tf.distribute does not distinguish whether the model is saved for inference
  # or for training. The implications are that all issues with serving are
  # issues with training as well, possibly with different symptoms.
  #
  # Note that for Keras models, loading them with tf.keras.models.load_model()
  # can workaround most issues since Keras loader restructs the layers with
  # saved configs if possible, in which case the saved graph is not used.

  def test_read_sync_on_read_variable_broken(self, strategy):
    # Reading a synchronizaiton=ON_READ in the replica context should only read
    # the local value, however with a loaded model, reading in the replica
    # context triggers aggregation as well. While one may argue the behavior is
    # desirable, note that aggregation can cause hanging if the originall model
    # is trained with MultiWorkerMirroredStrategy.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0.,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.SUM)

      @tf.function(input_signature=[])
      def __call__(self):
        return self.v.read_value()

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      m.v.assign(1.)
      self.assertAllEqual(
          self.evaluate(strategy.experimental_local_results(m.v)), [0.5, 0.5])
      tf.saved_model.save(m, export_dir)

    with strategy.scope():
      loaded = tf.saved_model.load(export_dir)
    # After loading, reading in the replica context is the same as reading in
    # the cross-replica context.
    # TODO(b/159752793): change after fix.
    self.assertAllEqual(
        self.evaluate(
            strategy.experimental_local_results(strategy.run(loaded))),
        [1., 1.])
    self.assertEqual(self.evaluate(loaded.v.read_value()), 1.)

  def test_update_sync_on_read_variable_broken(self, strategy):
    # Can't even save.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0.,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.SUM)

      @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
      def update(self, value):
        self.v.assign_add(value)

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      # got error, want no error.
      with self.assertRaisesRegex(ValueError,
                                  "SyncOnReadVariable does not support"):
        tf.saved_model.save(m, export_dir)

    # TODO(b/159752793): Complete the test after the saving issue is fixed.

  def test_read_mirrored_variable(self, strategy):

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0., synchronization=tf.VariableSynchronization.ON_WRITE)

      @tf.function(input_signature=[])
      def __call__(self):
        return self.v.read_value()

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      m.v.assign(1.)
      tf.saved_model.save(m, export_dir)

    with strategy.scope():
      loaded = tf.saved_model.load(export_dir)
    self.assertAllEqual(
        self.evaluate(
            strategy.experimental_local_results(strategy.run(loaded))),
        [1., 1.])

  def test_update_mirrored_variable_broken(self, strategy):
    # This is also uncommon since most model parameters should be updated by
    # optimizer, and this test case is for completeness.
    #
    # It's broken the saved model may not contain the aggregation logic. Even if
    # it does, it's wrong since all inputs to the aggregation are the same
    # variable.

    class Model(tf.Module):

      def __init__(self):
        self.v = tf.Variable(
            0.,
            synchronization=tf.VariableSynchronization.ON_WRITE,
            aggregation=tf.VariableAggregation.MEAN)

      @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
      def update(self, value):
        self.v.assign_add(value[0])

    export_dir = self.get_temp_dir()
    with strategy.scope():
      m = Model()
      tf.saved_model.save(m, export_dir)

    with strategy.scope():
      loaded = tf.saved_model.load(export_dir)
    value = strategy.experimental_distribute_dataset(
        tf.data.Dataset.from_tensor_slices([1., 2.]).batch(2))
    strategy.run(loaded.update, args=(next(iter(value)),))
    # TODO(b/159752793): Change after fix.
    # got [2., 4.], want [1.5, 1.5].
    self.assertAllEqual(
        self.evaluate(strategy.experimental_local_results(loaded.v)), [2., 4.])

  # TODO(crccw): add a test case that trains a saved model with optimizer.

  def test_model_with_loaded_v1_layer_broken(self, strategy):
    # If a model contains a layer loaded from SavedModel, and if the model is
    # loaded under tf.distribute.Strategy, it can't be saved and loaded again
    # under tf.distribute.Strategy.
    #
    # Although the error is the same models with TF2 SavedModel, the cause is
    # different. TF1 models loaded in API contain an intializer, which is
    # invoked upon loading. Since loading is in the cross-replica context, that
    # fails.
    #
    # Note that these saved model can still be loaded and used without
    # tf.distribute.Strategy.
    #
    # Some tf.hub layers are converted from TF1, by loading TF1 saved model in
    # TF2 then saved in TF2. This issue disables them to work with
    # tf.distribute.Strategy.
    v1_export_dir = self.get_temp_dir()

    with tf1.Graph().as_default(), tf1.Session() as sess:
      v = tf1.Variable(1., use_resource=True)
      sess.run(v.initializer)
      builder = tf1.saved_model.Builder(v1_export_dir)
      builder.add_meta_graph_and_variables(
          sess,
          tags=[tf1.saved_model.tag_constants.TRAINING],
          main_op=tf1.tables_initializer(),
          strip_default_attrs=True)
      builder.save()

    v1_loaded = tf.saved_model.load(v1_export_dir)
    v2_export_dir = self.get_temp_dir()
    tf.saved_model.save(v1_loaded, v2_export_dir)
    with strategy.scope():
      # TODO(b/150009657): remove after fix.
      # got error, want no error.
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          "from the cross-replica context in an in-replica context"):
        tf.saved_model.load(v2_export_dir)


if __name__ == "__main__":
  combinations.main()
