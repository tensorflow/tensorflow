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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf


class MiniModel(tf.keras.Model):
  """Minimal model for mnist.

  Useful for testing and debugging on slow TPU simulators.
  """

  def __init__(self):
    super(MiniModel, self).__init__(name='')
    self.fc = tf.keras.layers.Dense(1, name='fc', kernel_initializer='ones',
                                    bias_initializer='ones')

  def call(self, inputs, training=True):
    return self.fc(inputs)


class DefunnedMiniModel(MiniModel):

  @tf.function
  def call(self, inputs, training=True):
    return super(DefunnedMiniModel, self).call(inputs, training=training)


class ModelWithOptimizer(tf.keras.Model):

  def __init__(self):
    super(ModelWithOptimizer, self).__init__()
    self.dense = tf.keras.layers.Dense(1)
    self.optimizer = tf.keras.optimizers.Adam(0.01)

  @tf.function(
      input_signature=(tf.TensorSpec([None, 2], tf.float32),
                       tf.TensorSpec([None], tf.float32)))
  def call(self, x, y):
    with tf.GradientTape() as tape:
      loss = tf.math.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {'loss': loss}


class FunctionTest(tf.test.TestCase):

  def testFunctionRelaxationLosesInnerDimWithKerasLayer(self):
    layer = tf.keras.layers.Dense(1)
    fn = tf.function(experimental_relax_shapes=True)(layer)

    with self.captureWritesToStream(sys.stderr) as printed:
      fn(tf.ones((3, 2)))
      self.assertNotIn('ValueError', printed.contents())
    with self.captureWritesToStream(sys.stderr) as printed:
      # Use batch size 2 to trigger a second cache miss on the shape.
      fn(tf.ones((2, 2)))
      self.assertNotIn('ValueError', printed.contents())

    # Shape relaxation passes TensorShape([None, None]), which causes layer
    # matmul to fail, due to incompatible dims.  What would have been a graph
    # build time error (layer would complain about the inner dim being 4).
    with self.captureWritesToStream(sys.stderr) as printed:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'Matrix size-incompatible'):
        fn(tf.ones((3, 4)))

  def testDefunKerasModelCall(self):
    model = MiniModel()
    model.call = tf.function(model.call)

    x = tf.ones([1, 2])
    y = model(x)  # pylint:disable=not-callable

    self.assertAllEqual([[3.0]], self.evaluate(y))

    # Break the reference cycle between the MiniModel and the defun:
    # `MiniModel` --(through its `call` method)--> `Function`
    # `Function` --(instancemethod on `MiniModel`)--> `MiniModel`
    del model.call

  def testDecoratedMethod(self):
    m = DefunnedMiniModel()
    instance_call_one = m.call(tf.ones([1, 2]), training=True)
    instance_call_two = m.call(
        inputs=tf.ones([1, 2]), training=True)
    class_call = DefunnedMiniModel.call(m, tf.ones([1, 2]), training=True)
    self.assertAllEqual(instance_call_one, instance_call_two)
    self.assertAllEqual(instance_call_one, class_call)

  def testDecoratedMethodUniqueFunctionPerInstance(self):
    m = DefunnedMiniModel()
    n = DefunnedMiniModel()

    class_method_one = DefunnedMiniModel.call
    class_method_two = DefunnedMiniModel.call

    m_method_one = m.call
    m_method_two = m.call

    n_method_one = n.call
    n_method_two = n.call

    self.assertEqual(class_method_one, class_method_two)
    self.assertEqual(m_method_one, m_method_two)
    self.assertEqual(n_method_one, n_method_two)
    self.assertNotEqual(m.call, n.call)

  def testDecoratedMethodGetConcreteFunction(self):
    m = DefunnedMiniModel()
    instance_call_one = m.call.get_concrete_function(
        tf.ones([1, 2]), training=False)
    instance_call_two = m.call.get_concrete_function(
        inputs=tf.ones([1, 2]), training=False)
    self.assertAllEqual(instance_call_one(tf.ones([1, 2])),
                        instance_call_two(tf.ones([1, 2])))

    # Also make sure get_concrete_function works on the class method
    DefunnedMiniModel.call.get_concrete_function(
        m, tf.ones([1, 2]), training=False)
    DefunnedMiniModel.call.get_concrete_function(
        m, inputs=tf.ones([1, 2]), training=True)

  def testDecoratedMethodVariableCleanup(self):
    m = DefunnedMiniModel()
    m(tf.ones([1, 2]))  # pylint:disable=not-callable
    variable_refs = list({v.ref() for v in m.variables})
    self.assertLen(variable_refs, 2)
    del m

    # Verifying if the variables are only referenced from variable_refs.
    # We expect the reference counter to be 1, but `sys.getrefcount` reports
    # one higher reference counter because a temporary is created when we call
    # sys.getrefcount().  Hence check if the number returned is 2.
    # https://docs.python.org/3/library/sys.html#sys.getrefcount
    self.assertEqual(sys.getrefcount(variable_refs[0].deref()), 2)
    self.assertEqual(sys.getrefcount(variable_refs[1].deref()), 2)

  def testStandardTrainingLoopInFunction(self):
    layer = tf.keras.layers.Dense(2)
    dataset = (
        tf.data.Dataset.from_tensors((tf.ones([784]), tf.ones([], tf.int32)))
        .map(lambda x, y: (x, y))
        .repeat(10)
        .batch(32))
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train():
      for x, y in dataset:
        with tf.GradientTape() as tape:
          out = layer(x)
          loss = tf.reduce_mean(
              tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=out, labels=y))
        layer_variables = layer.trainable_variables
        gradients = tape.gradient(loss, layer_variables)
        optimizer.apply_gradients(zip(gradients, layer_variables))

    train()

  def testEarlyStoppingTrainingLoopInFunction(self):
    layer = tf.keras.layers.Dense(2)
    dataset = (
        tf.data.Dataset.from_tensors((tf.ones([784]), tf.ones([], tf.int32)))
        .map(lambda x, y: (x, y))
        .repeat(10)
        .batch(32))
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train():
      for x, y in dataset:
        with tf.GradientTape() as tape:
          out = layer(x)
          loss = tf.math.reduce_mean(
              tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=out, labels=y))
        layer_variables = layer.trainable_variables
        gradients = tape.gradient(loss, layer_variables)
        optimizer.apply_gradients(zip(gradients, layer_variables))
        if optimizer.iterations > 3:
          break

    train()

  def test_optimizer(self):
    x = tf.constant([[3., 4.]])
    y = tf.constant([2.])
    model = ModelWithOptimizer()
    model(x, y)  # pylint:disable=not-callable


class AutomaticControlDependenciesTest(tf.test.TestCase):

  def testVariableInitializersCanBeLifted(self):
    # The initializer is a stateful op, but using it inside a function should
    # *not* create additional dependencies.  That's what we're testing.
    layer = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

    @tf.function
    def fn(x):
      # Stateful operation
      tf.debugging.Assert(x, ['Error'])
      # Variable initialization should be lifted.  Prior to the change that
      # added this test, the lifting would crash because of an auto control dep
      # added on `x`.  Note, the error did not happen if we
      # manually created a tf.Variable outside of function and used it
      # here.  Alternatively, creating a tf.Variable inside fn() causes
      # a different sort of error that is out of scope for this test.
      return layer(tf.convert_to_tensor([[1.0, 1.0]]))

    true = tf.convert_to_tensor(True)

    concrete = fn.get_concrete_function(
        tf.TensorSpec(shape=(), dtype=tf.bool))
    self.evaluate(concrete(true))
    self.evaluate(fn(True))


if __name__ == '__main__':
  tf.test.main()
