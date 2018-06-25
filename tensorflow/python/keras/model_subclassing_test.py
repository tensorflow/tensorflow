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
"""Tests for Model subclassing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.rmsprop import RMSPropOptimizer

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


# pylint: disable=not-callable
class SimpleTestModel(keras.Model):

  def __init__(self, use_bn=False, use_dp=False, num_classes=10):
    super(SimpleTestModel, self).__init__(name='test_model')
    self.use_bn = use_bn
    self.use_dp = use_dp
    self.num_classes = num_classes

    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
    if self.use_dp:
      self.dp = keras.layers.Dropout(0.5)
    if self.use_bn:
      self.bn = keras.layers.BatchNormalization(axis=-1)

  def call(self, x):
    x = self.dense1(x)
    if self.use_dp:
      x = self.dp(x)
    if self.use_bn:
      x = self.bn(x)
    return self.dense2(x)


class MultiIOTestModel(keras.Model):

  def __init__(self, use_bn=False, use_dp=False, num_classes=(2, 3)):
    super(MultiIOTestModel, self).__init__(name='test_model')
    self.use_bn = use_bn
    self.use_dp = use_dp
    self.num_classes = num_classes

    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes[0], activation='softmax')
    self.dense3 = keras.layers.Dense(num_classes[1], activation='softmax')
    if use_dp:
      self.dp = keras.layers.Dropout(0.5)
    if use_bn:
      self.bn = keras.layers.BatchNormalization()

  def call(self, inputs):
    x1, x2 = inputs
    x1 = self.dense1(x1)
    x2 = self.dense1(x2)
    if self.use_dp:
      x1 = self.dp(x1)
    if self.use_bn:
      x2 = self.bn(x2)
    return [self.dense2(x1), self.dense3(x2)]


class NestedTestModel1(keras.Model):
  """A model subclass nested inside a model subclass.
  """

  def __init__(self, num_classes=2):
    super(NestedTestModel1, self).__init__(name='nested_model_1')
    self.num_classes = num_classes
    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes, activation='relu')
    self.bn = keras.layers.BatchNormalization()
    self.test_net = SimpleTestModel(num_classes=4,
                                    use_bn=True,
                                    use_dp=True)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.bn(x)
    x = self.test_net(x)
    return self.dense2(x)


def get_functional_graph_model(input_dim, num_classes):
  # A simple functional-API model (a.k.a. graph network)
  inputs = keras.Input(shape=(input_dim,))
  x = keras.layers.Dense(32, activation='relu')(inputs)
  x = keras.layers.BatchNormalization()(x)
  outputs = keras.layers.Dense(num_classes)(x)
  return keras.Model(inputs, outputs)


class NestedTestModel2(keras.Model):
  """A model subclass with a functional-API graph network inside.
  """

  def __init__(self, num_classes=2):
    super(NestedTestModel2, self).__init__(name='nested_model_2')
    self.num_classes = num_classes
    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes, activation='relu')
    self.bn = self.bn = keras.layers.BatchNormalization()
    self.test_net = get_functional_graph_model(32, 4)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.bn(x)
    x = self.test_net(x)
    return self.dense2(x)


def get_nested_model_3(input_dim, num_classes):
  # A functional-API model with a subclassed model inside.
  # NOTE: this requires the inner subclass to implement `compute_output_shape`.

  inputs = keras.Input(shape=(input_dim,))
  x = keras.layers.Dense(32, activation='relu')(inputs)
  x = keras.layers.BatchNormalization()(x)

  class Inner(keras.Model):

    def __init__(self):
      super(Inner, self).__init__()
      self.dense1 = keras.layers.Dense(32, activation='relu')
      self.dense2 = keras.layers.Dense(5, activation='relu')
      self.bn = keras.layers.BatchNormalization()

    def call(self, inputs):
      x = self.dense1(inputs)
      x = self.dense2(x)
      return self.bn(x)

    def compute_output_shape(self, input_shape):
      return tensor_shape.TensorShape((input_shape[0], 5))

  test_model = Inner()
  x = test_model(x)
  outputs = keras.layers.Dense(num_classes)(x)
  return keras.Model(inputs, outputs, name='nested_model_3')


class ModelSubclassingTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_single_io_workflow_with_np_arrays(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = SimpleTestModel(num_classes=num_classes,
                            use_dp=True,
                            use_bn=True)
    model.compile(loss='mse',
                  optimizer=RMSPropOptimizer(learning_rate=0.001),
                  metrics=['acc'])

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

  @test_util.run_in_graph_and_eager_modes
  def test_multi_io_workflow_with_np_arrays(self):
    num_classes = (2, 3)
    num_samples = 1000
    input_dim = 50

    model = MultiIOTestModel(num_classes=num_classes,
                             use_dp=True,
                             use_bn=True)
    model.compile(loss='mse',
                  optimizer=RMSPropOptimizer(learning_rate=0.001),
                  metrics=['acc'])

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate([x1, x2], [y1, y2], verbose=0)

  def test_single_io_workflow_with_tensors(self):

    num_classes = 2
    num_samples = 10
    input_dim = 50

    with self.test_session():
      model = SimpleTestModel(num_classes=num_classes,
                              use_dp=True,
                              use_bn=True)
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))

      x = array_ops.ones((num_samples, input_dim))
      y = array_ops.zeros((num_samples, num_classes))

      model.fit(x, y, epochs=2, steps_per_epoch=10, verbose=0)
      _ = model.evaluate(steps=10, verbose=0)

  def test_multi_io_workflow_with_tensors(self):

    num_classes = (2, 3)
    num_samples = 10
    input_dim = 50

    with self.test_session():
      model = MultiIOTestModel(num_classes=num_classes,
                               use_dp=True,
                               use_bn=True)
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))

      x1 = array_ops.ones((num_samples, input_dim))
      x2 = array_ops.ones((num_samples, input_dim))
      y1 = array_ops.zeros((num_samples, num_classes[0]))
      y2 = array_ops.zeros((num_samples, num_classes[1]))

      model.fit([x1, x2], [y1, y2], epochs=2, steps_per_epoch=10, verbose=0)
      _ = model.evaluate(steps=10, verbose=0)

  @test_util.run_in_graph_and_eager_modes
  def test_single_io_workflow_with_dataset_iterators(self):
    num_classes = 2
    num_samples = 10
    input_dim = 50

    with self.test_session():
      model = SimpleTestModel(num_classes=num_classes, use_dp=True, use_bn=True)
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))

      x = np.ones((num_samples, input_dim))
      y = np.zeros((num_samples, num_classes))
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)
      iterator = dataset.make_one_shot_iterator()

      model.fit(iterator, epochs=2, steps_per_epoch=10, verbose=0)
      _ = model.evaluate(iterator, steps=10, verbose=0)

  def test_multi_io_workflow_with_numpy_arrays_and_custom_placeholders(self):

    num_classes = (2, 3)
    num_samples = 1000
    input_dim = 50

    with self.test_session():
      model = MultiIOTestModel(num_classes=num_classes,
                               use_dp=True,
                               use_bn=True)
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))

      x1 = np.ones((num_samples, input_dim))
      x2 = np.ones((num_samples, input_dim))
      y1 = np.zeros((num_samples, num_classes[0]))
      y2 = np.zeros((num_samples, num_classes[1]))

      x2_placeholder = array_ops.placeholder(
          dtype='float32', shape=(None, input_dim))
      model._set_inputs([x1, x2_placeholder])

      model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
      _ = model.evaluate([x1, x2], [y1, y2], verbose=0)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def test_attributes(self):
    # layers, weights, trainable_weights, non_trainable_weights, inputs, outputs

    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    self.assertEqual(model.name, 'test_model')
    self.assertEqual(model.built, False)
    self.assertEqual(len(model.weights), 0)

    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    model.train_on_batch([x1, x2], [y1, y2])

    self.assertEqual(model.built, True)
    self.assertEqual(len(model.layers), 4)
    self.assertEqual(len(model.weights), 10)
    self.assertEqual(len(model.trainable_weights), 8)
    self.assertEqual(len(model.non_trainable_weights), 2)
    self.assertEqual(len(model.inputs), 2)
    self.assertEqual(len(model.outputs), 2)

  @test_util.run_in_graph_and_eager_modes
  def test_updates(self):
    # test that updates get run during training
    num_samples = 100
    input_dim = 50

    class BNNet(keras.Model):

      def __init__(self):
        super(BNNet, self).__init__()
        self.bn = keras.layers.BatchNormalization(beta_initializer='ones',
                                                  gamma_initializer='ones')

      def call(self, inputs):
        return self.bn(inputs)

    x = np.ones((num_samples, input_dim))
    y = np.ones((num_samples, input_dim))

    model = BNNet()
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    y_ref = model.predict(x)

    model.train_on_batch(x, y)
    y_new = model.predict(x)
    self.assertGreater(np.sum(np.abs(y_ref - y_new)), 0.1)

  def test_updates_and_losses_for_nested_models_in_subclassed_model(self):

    # Case 1: deferred-build sequential nested in subclass.
    class TestModel1(keras.Model):

      def __init__(self):
        super(TestModel1, self).__init__()
        self.fc = keras.layers.Dense(10, input_shape=(784,),
                                     activity_regularizer='l1')
        self.bn = keras.Sequential([keras.layers.BatchNormalization(axis=1)])

      def call(self, x):
        return self.bn(self.fc(x))

    with self.test_session():
      model = TestModel1()

      x = array_ops.ones(shape=[100, 784], dtype='float32')
      model(x)
      self.assertEqual(len(model.get_updates_for(x)), 2)
      self.assertEqual(len(model.get_losses_for(x)), 1)

    # Case 2: placeholder-sequential nested in subclass.
    class TestModel2(keras.Model):

      def __init__(self):
        super(TestModel2, self).__init__()
        self.fc = keras.layers.Dense(10, input_shape=(784,),
                                     activity_regularizer='l1')
        self.bn = keras.Sequential(
            [keras.layers.BatchNormalization(axis=1, input_shape=(10,))])

      def call(self, x):
        return self.bn(self.fc(x))

    with self.test_session():
      model = TestModel2()

      x = array_ops.ones(shape=[100, 784], dtype='float32')
      model(x)
      self.assertEqual(len(model.get_updates_for(x)), 2)
      self.assertEqual(len(model.get_losses_for(x)), 1)

    # Case 3: functional-API model nested in subclass.
    inputs = keras.Input((10,))
    outputs = keras.layers.BatchNormalization(axis=1)(inputs)
    bn = keras.Model(inputs, outputs)

    class TestModel3(keras.Model):

      def __init__(self):
        super(TestModel3, self).__init__()
        self.fc = keras.layers.Dense(10, input_shape=(784,),
                                     activity_regularizer='l1')
        self.bn = bn

      def call(self, x):
        return self.bn(self.fc(x))

    with self.test_session():
      model = TestModel3()

      x = array_ops.ones(shape=[100, 784], dtype='float32')
      model(x)
      self.assertEqual(len(model.get_updates_for(x)), 2)
      self.assertEqual(len(model.get_losses_for(x)), 1)

  @test_util.run_in_graph_and_eager_modes
  def test_training_and_inference_behavior(self):
    # test that dropout is applied in training and not inference

    num_samples = 100
    input_dim = 50

    class DPNet(keras.Model):

      def __init__(self):
        super(DPNet, self).__init__()
        self.dp = keras.layers.Dropout(0.5)
        self.dense = keras.layers.Dense(1,
                                        use_bias=False,
                                        kernel_initializer='ones')

      def call(self, inputs):
        x = self.dp(inputs)
        return self.dense(x)

    model = DPNet()
    x = np.ones((num_samples, input_dim))
    y = model.predict(x)
    self.assertEqual(np.sum(y), np.sum(x))
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    loss = model.train_on_batch(x, y)
    self.assertGreater(loss, 0.1)

  @test_util.run_in_graph_and_eager_modes
  def test_training_methods(self):
    # test fit, train_on_batch
    # on different input types: list, dict

    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
    model.fit({'input_1': x1, 'input_2': x2},
              {'output_1': y1, 'output_2': y2},
              epochs=2, batch_size=32)
    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0,
              validation_data=([x1, x2], [y1, y2]))

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    model.train_on_batch([x1, x2], [y1, y2])
    model.train_on_batch({'input_1': x1, 'input_2': x2},
                         {'output_1': y1, 'output_2': y2})

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def test_inference_methods(self):
    # test predict, evaluate, test_on_batch, predict_on_batch
    # on different input types: list, dict
    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    model.evaluate([x1, x2], [y1, y2])
    model.test_on_batch([x1, x2], [y1, y2])

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)
    model.predict([x1, x2])

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)
    model.predict_on_batch([x1, x2])

  @test_util.run_in_graph_and_eager_modes
  def test_trainable_mutation(self):
    # test that you can change `trainable` on a model or layer, and that
    # it freezes the model state during training
    # TODO(fchollet): add test after we unify BN behavior in eager and symbolic.
    pass

  @test_util.run_in_graph_and_eager_modes
  def test_saving(self):

    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
    y_ref_1, y_ref_2 = model.predict([x1, x2])

    tf_format_name = os.path.join(self.get_temp_dir(), 'ckpt')
    model.save_weights(tf_format_name)
    if h5py is not None:
      hdf5_format_name = os.path.join(self.get_temp_dir(), 'weights.h5')
      model.save_weights(hdf5_format_name)

    model = MultiIOTestModel(num_classes=num_classes, use_bn=True)

    if h5py is not None:
      with self.assertRaises(ValueError):
        model.load_weights(hdf5_format_name)

    model.load_weights(tf_format_name)

    y1, y2 = model.predict([x1, x2])
    self.assertAllClose(y_ref_1, y1, atol=1e-5)
    self.assertAllClose(y_ref_2, y2, atol=1e-5)

    if h5py is not None:
      model.load_weights(hdf5_format_name)

      y1, y2 = model.predict([x1, x2])
      self.assertAllClose(y_ref_1, y1, atol=1e-5)
      self.assertAllClose(y_ref_2, y2, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_summary(self):

    class ToString(object):

      def __init__(self):
        self.contents = ''

      def __call__(self, msg):
        self.contents += msg + '\n'

    # Single-io
    model = SimpleTestModel(num_classes=4, use_bn=True, use_dp=True)
    model._set_inputs(np.ones((3, 4)))  # need to build model first
    print_fn = ToString()
    model.summary(print_fn=print_fn)
    self.assertTrue('Trainable params: 356' in print_fn.contents)

    # Multi-io
    model = MultiIOTestModel(num_classes=(5, 6), use_bn=True, use_dp=True)
    model._set_inputs([np.ones((3, 4)),
                       np.ones((3, 4))])  # need to build model first
    print_fn = ToString()
    model.summary(print_fn=print_fn)
    self.assertTrue('Trainable params: 587' in print_fn.contents)

  @test_util.run_in_graph_and_eager_modes
  def test_subclass_nested_in_subclass(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = NestedTestModel1(num_classes=num_classes)
    model.compile(loss='mse',
                  optimizer=RMSPropOptimizer(learning_rate=0.001),
                  metrics=['acc'])

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 8 + len(model.test_net.weights))
    self.assertEqual(len(model.non_trainable_weights),
                     2 + len(model.test_net.non_trainable_weights))
    self.assertEqual(len(model.trainable_weights),
                     6 + len(model.test_net.trainable_weights))

  @test_util.run_in_graph_and_eager_modes
  def test_graph_nested_in_subclass(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = NestedTestModel2(num_classes=num_classes)
    model.compile(loss='mse',
                  optimizer=RMSPropOptimizer(learning_rate=0.001),
                  metrics=['acc'])

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 8 + len(model.test_net.weights))
    self.assertEqual(len(model.non_trainable_weights),
                     2 + len(model.test_net.non_trainable_weights))
    self.assertEqual(len(model.trainable_weights),
                     6 + len(model.test_net.trainable_weights))

  @test_util.run_in_graph_and_eager_modes
  def test_subclass_nested_in_graph(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = get_nested_model_3(input_dim=input_dim, num_classes=num_classes)
    model.compile(loss='mse',
                  optimizer=RMSPropOptimizer(learning_rate=0.001),
                  metrics=['acc'])

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 16)
    self.assertEqual(
        len(model.non_trainable_weights), 4)
    self.assertEqual(len(model.trainable_weights), 12)

  @test_util.run_in_graph_and_eager_modes
  def test_support_for_manual_training_arg(self):
    # In most cases, the `training` argument is left unspecified, in which
    # case it defaults to value corresponding to the Model method being used
    # (fit -> True, predict -> False, etc).
    # If the user writes their model `call` method to take
    # an explicit `training` argument, we must check that the correct value
    # is being passed to the model for each method call.

    class DPNet(keras.Model):

      def __init__(self):
        super(DPNet, self).__init__()
        self.dp = keras.layers.Dropout(0.5)
        self.dense = keras.layers.Dense(1,
                                        use_bias=False,
                                        kernel_initializer='ones')

      def call(self, inputs, training=False):
        x = self.dp(inputs, training=training)
        return self.dense(x)

    model = DPNet()
    x = np.ones((10, 10))
    y = model.predict(x)
    self.assertEqual(np.sum(y), np.sum(x))
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    loss = model.train_on_batch(x, y)
    self.assertGreater(loss, 0.1)

  def test_no_dependency(self):
    class Foo(keras.Model):

      def __init__(self):
        super(Foo, self).__init__()
        self.isdep = keras.layers.Dense(1)
        self.notdep = checkpointable.NoDependency(keras.layers.Dense(2))
        self.notdep_var = checkpointable.NoDependency(
            resource_variable_ops.ResourceVariable(1., name='notdep_var'))

    m = Foo()
    self.assertEqual([m.isdep, m.notdep], m.layers)
    self.assertEqual(1, len(m._checkpoint_dependencies))
    self.assertIs(m.isdep, m._checkpoint_dependencies[0].ref)
    self.assertEqual('notdep_var:0', m.notdep_var.name)

  def test_extra_variable(self):

    class ExtraVar(keras.Model):

      def __init__(self):
        super(ExtraVar, self).__init__()
        self.dense = keras.layers.Dense(1)
        self.var = resource_variable_ops.ResourceVariable(1.)
        self.not_trainable_var = resource_variable_ops.ResourceVariable(
            2., trainable=False)

      def call(self, inputs):
        return self.dense(inputs + self.var)

    m = ExtraVar()
    self.assertTrue(m.trainable)
    self.assertEqual([m.dense], m.layers)
    self.assertEqual([m.var, m.not_trainable_var], m.variables)
    self.assertEqual([m.var], m.trainable_variables)
    self.assertEqual([m.not_trainable_var], m.non_trainable_variables)
    m.trainable = False
    self.assertEqual([m.var, m.not_trainable_var], m.variables)
    self.assertEqual([], m.trainable_variables)
    self.assertEqual([m.var, m.not_trainable_var], m.non_trainable_variables)
    m.trainable = True

    m(array_ops.ones([1, 1]))

    self.assertEqual([m.dense.kernel, m.dense.bias], m.dense.variables)
    self.assertEqual([m.dense.kernel, m.dense.bias], m.dense.weights)

    self.assertEqual([m.dense.kernel, m.dense.bias, m.var, m.not_trainable_var],
                     m.variables)
    self.assertEqual([m.dense.kernel, m.dense.bias, m.var],
                     m.trainable_variables)
    self.assertEqual([m.not_trainable_var], m.non_trainable_variables)

    m.dense.trainable = False
    self.assertEqual(
        [m.var, m.dense.kernel, m.dense.bias, m.not_trainable_var],
        m.variables)
    self.assertEqual([m.var], m.trainable_variables)
    self.assertEqual([m.dense.kernel, m.dense.bias, m.not_trainable_var],
                     m.non_trainable_variables)


class CustomCallModel(keras.Model):

  def __init__(self):
    super(CustomCallModel, self).__init__()
    self.dense1 = keras.layers.Dense(1, activation='relu')
    self.dense2 = keras.layers.Dense(1, activation='softmax')

  def call(self, first, second, fiddle_with_output='no', training=True):
    combined = self.dense1(first) + self.dense2(second)
    if fiddle_with_output == 'yes':
      return 10. * combined
    else:
      return combined


class CustomCallSignatureTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_no_inputs_in_signature(self):
    model = CustomCallModel()
    first = array_ops.ones([2, 3])
    second = array_ops.ones([2, 5])
    output = model(first, second)
    self.evaluate([v.initializer for v in model.variables])
    expected_output = self.evaluate(model.dense1(first) + model.dense2(second))
    self.assertAllClose(expected_output, self.evaluate(output))
    output = model(first, second, fiddle_with_output='yes')
    self.assertAllClose(10. * expected_output, self.evaluate(output))
    output = model(first, second=second, training=False)
    self.assertAllClose(expected_output, self.evaluate(output))

  @test_util.run_in_graph_and_eager_modes
  def test_inputs_in_signature(self):

    class HasInputsAndOtherPositional(keras.Model):

      def call(self, inputs, some_other_arg, training=False):
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    model = HasInputsAndOtherPositional()
    with self.assertRaisesRegexp(
        TypeError, 'everything else as a keyword argument'):
      x1, x2 = keras.Input((1, 1)), keras.Input((1, 1))
      model(x1, x2)

  @test_util.run_in_graph_and_eager_modes
  def test_kwargs_in_signature(self):

    class HasKwargs(keras.Model):

      def call(self, x, y=3, **key_words):
        return x

    model = HasKwargs()
    arg = array_ops.ones([])
    model(arg, a=3)
    if not context.executing_eagerly():
      six.assertCountEqual(self, [arg], model.inputs)

  @test_util.run_in_graph_and_eager_modes
  def test_args_in_signature(self):

    class HasArgs(keras.Model):

      def call(self, x, *args, **kwargs):
        return [x] + list(args)

      def compute_output_shape(self, input_shape):
        return input_shape

    model = HasArgs()
    x1, x2, x3 = keras.Input((1, 1)), keras.Input((1, 1)), keras.Input((1, 1))
    model(x1, x2, x3, a=3)
    if not context.executing_eagerly():
      six.assertCountEqual(self, [x1, x2, x3], model.inputs)

  def test_args_and_keywords_in_signature(self):

    class HasArgs(keras.Model):

      def call(self, x, training=True, *args, **kwargs):
        return x

    with context.graph_mode():
      model = HasArgs()
      x1, x2, x3 = keras.Input((1, 1)), keras.Input((1, 1)), keras.Input((1, 1))
      with self.assertRaisesRegexp(TypeError, 'args and arguments with'):
        model(x1, x2, x3, a=3)

  def test_training_no_default(self):

    class TrainingNoDefault(keras.Model):

      def call(self, x, training):
        return x

    with context.graph_mode():
      model = TrainingNoDefault()
      arg = array_ops.ones([])
      model(arg, True)
      six.assertCountEqual(self, [arg], model.inputs)

  def test_training_no_default_with_positional(self):

    class TrainingNoDefaultWithPositional(keras.Model):

      def call(self, x, training, positional):
        return x

    with context.graph_mode():
      model = TrainingNoDefaultWithPositional()
      x1, x2, x3 = keras.Input((1, 1)), keras.Input((1, 1)), keras.Input((1, 1))
      with self.assertRaisesRegexp(TypeError, 'after a non-input'):
        model(x1, x2, x3)

if __name__ == '__main__':
  test.main()
