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
#,============================================================================
"""Tests for layer graphs construction & handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import combinations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer as input_layer_lib
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow.python.training.tracking.util import Checkpoint

try:
  import yaml  # pylint:disable=g-import-not-at-top
except ImportError:
  yaml = None


class NetworkConstructionTest(keras_parameterized.TestCase):

  def test_get_updates(self):

    class MyLayer(layers.Layer):

      def build(self, input_shape):
        self.a = self.add_variable('a',
                                   (1, 1),
                                   'float32',
                                   trainable=False)
        self.b = self.add_variable('b',
                                   (1, 1),
                                   'float32',
                                   trainable=False)
        self.add_update(state_ops.assign_add(self.a, [[1.]],
                                             name='unconditional_update'))
        self.built = True

      def call(self, inputs):
        self.add_update(state_ops.assign_add(self.b, inputs,
                                             name='conditional_update'),
                        inputs=True)
        return inputs + 1

    with ops.Graph().as_default():
      x1 = input_layer_lib.Input(shape=(1,))
      layer = MyLayer()
      _ = layer(x1)

      self.assertEqual(len(layer.updates), 2)

      x2 = input_layer_lib.Input(shape=(1,))
      y2 = layer(x2)

      self.assertEqual(len(layer.updates), 3)

      network = functional.Functional(x2, y2)
      self.assertEqual(len(network.updates), 3)

      x3 = input_layer_lib.Input(shape=(1,))
      _ = layer(x3)
      self.assertEqual(len(network.updates), 4)

      x4 = input_layer_lib.Input(shape=(1,))
      _ = network(x4)
      self.assertEqual(len(network.updates), 5)

      network.add_update(state_ops.assign_add(layer.a, [[1]]))
      self.assertEqual(len(network.updates), 6)

      network.add_update(state_ops.assign_add(layer.b, x4), inputs=True)
      self.assertEqual(len(network.updates), 7)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_get_updates_bn(self):
    x1 = input_layer_lib.Input(shape=(1,))
    layer = layers.BatchNormalization()
    _ = layer(x1)

    self.assertEqual(len(layer.updates), 2)

  def test_get_layer(self):
    # create a simple network
    x = input_layer_lib.Input(shape=(32,))
    dense_a = layers.Dense(4, name='dense_a')
    dense_b = layers.Dense(2, name='dense_b')
    y = dense_b(dense_a(x))
    network = functional.Functional(x, y, name='dense_network')

    # test various get_layer by index
    self.assertEqual(network.get_layer(index=1), dense_a)

    # test invalid get_layer by index
    with self.assertRaisesRegexp(
        ValueError, 'Was asked to retrieve layer at index ' + str(3) +
        ' but model only has ' + str(len(network.layers)) + ' layers.'):
      network.get_layer(index=3)

    # test that only one between name and index is requested
    with self.assertRaisesRegexp(ValueError,
                                 'Provide only a layer name or a layer index'):
      network.get_layer(index=1, name='dense_b')

    # test that a name or an index must be provided
    with self.assertRaisesRegexp(ValueError,
                                 'Provide either a layer name or layer index.'):
      network.get_layer()

    # test various get_layer by name
    self.assertEqual(network.get_layer(name='dense_a'), dense_a)

    # test invalid get_layer by name
    with self.assertRaisesRegexp(ValueError, 'No such layer: dense_c.'):
      network.get_layer(name='dense_c')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTopologicalAttributes(self):
    # test layer attributes / methods related to cross-layer connectivity.
    a = input_layer_lib.Input(shape=(32,), name='input_a')
    b = input_layer_lib.Input(shape=(32,), name='input_b')

    # test input, output, input_shape, output_shape
    test_layer = layers.Dense(16, name='test_layer')
    a_test = test_layer(a)
    self.assertIs(test_layer.input, a)
    self.assertIs(test_layer.output, a_test)
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, (None, 16))

    # test `get_*_at` methods
    dense = layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    self.assertIs(dense.get_input_at(0), a)
    self.assertIs(dense.get_input_at(1), b)
    self.assertIs(dense.get_output_at(0), a_2)
    self.assertIs(dense.get_output_at(1), b_2)
    self.assertEqual(dense.get_input_shape_at(0), (None, 32))
    self.assertEqual(dense.get_input_shape_at(1), (None, 32))
    self.assertEqual(dense.get_output_shape_at(0), (None, 16))
    self.assertEqual(dense.get_output_shape_at(1), (None, 16))

    # Test invalid value for attribute retrieval.
    with self.assertRaises(ValueError):
      dense.get_input_at(2)
    with self.assertRaises(AttributeError):
      new_dense = layers.Dense(16)
      _ = new_dense.input
    with self.assertRaises(AttributeError):
      new_dense = layers.Dense(16)
      _ = new_dense.output
    with self.assertRaises(AttributeError):
      new_dense = layers.Dense(16)
      _ = new_dense.output_shape
    with self.assertRaises(AttributeError):
      new_dense = layers.Dense(16)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = layers.Dense(16)
      a = input_layer_lib.Input(shape=(3, 32))
      a = input_layer_lib.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = layers.Dense(16)
      a = input_layer_lib.Input(shape=(3, 32))
      a = input_layer_lib.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.output_shape

  def _assertAllIs(self, a, b):
    self.assertTrue(all(x is y for x, y in zip(a, b)))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTopologicalAttributesMultiOutputLayer(self):

    class PowersLayer(layers.Layer):

      def call(self, inputs):
        return [inputs**2, inputs**3]

    x = input_layer_lib.Input(shape=(32,))
    test_layer = PowersLayer()
    p1, p2 = test_layer(x)  # pylint: disable=not-callable

    self.assertIs(test_layer.input, x)
    self._assertAllIs(test_layer.output, [p1, p2])
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, [(None, 32), (None, 32)])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTopologicalAttributesMultiInputLayer(self):

    class AddLayer(layers.Layer):

      def call(self, inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]

    a = input_layer_lib.Input(shape=(32,))
    b = input_layer_lib.Input(shape=(32,))
    test_layer = AddLayer()
    y = test_layer([a, b])  # pylint: disable=not-callable

    self._assertAllIs(test_layer.input, [a, b])
    self.assertIs(test_layer.output, y)
    self.assertEqual(test_layer.input_shape, [(None, 32), (None, 32)])
    self.assertEqual(test_layer.output_shape, (None, 32))

  def testBasicNetwork(self):
    with ops.Graph().as_default():
      # minimum viable network
      x = input_layer_lib.Input(shape=(32,))
      dense = layers.Dense(2)
      y = dense(x)
      network = functional.Functional(x, y, name='dense_network')

      # test basic attributes
      self.assertEqual(network.name, 'dense_network')
      self.assertEqual(len(network.layers), 2)  # InputLayer + Dense
      self.assertEqual(network.layers[1], dense)
      self._assertAllIs(network.weights, dense.weights)
      self._assertAllIs(network.trainable_weights, dense.trainable_weights)
      self._assertAllIs(network.non_trainable_weights,
                        dense.non_trainable_weights)

      # test callability on Input
      x_2 = input_layer_lib.Input(shape=(32,))
      y_2 = network(x_2)
      self.assertEqual(y_2.shape.as_list(), [None, 2])

      # test callability on regular tensor
      x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
      y_2 = network(x_2)
      self.assertEqual(y_2.shape.as_list(), [None, 2])

      # test network `trainable` attribute
      network.trainable = False
      self._assertAllIs(network.weights, dense.weights)
      self.assertEqual(network.trainable_weights, [])
      self._assertAllIs(network.non_trainable_weights,
                        dense.trainable_weights + dense.non_trainable_weights)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_trainable_weights(self):
    a = layers.Input(shape=(2,))
    b = layers.Dense(1)(a)
    model = training_lib.Model(a, b)

    weights = model.weights
    self._assertAllIs(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self._assertAllIs(model.non_trainable_weights, weights)

    model.trainable = True
    self._assertAllIs(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.layers[1].trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self._assertAllIs(model.non_trainable_weights, weights)

    # sequential model
    model = sequential.Sequential()
    model.add(layers.Dense(1, input_dim=2))
    weights = model.weights

    self._assertAllIs(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self._assertAllIs(model.non_trainable_weights, weights)

    model.trainable = True
    self._assertAllIs(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.layers[0].trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self._assertAllIs(model.non_trainable_weights, weights)

  def test_layer_call_arguments(self):
    with ops.Graph().as_default():
      # Test the ability to pass and serialize arguments to `call`.
      inp = layers.Input(shape=(2,))
      x = layers.Dense(3)(inp)
      x = layers.Dropout(0.5)(x, training=True)
      model = training_lib.Model(inp, x)
      # Would be `dropout/cond/Merge` by default
      self.assertIn('dropout', model.output.op.name)

      # Test that argument is kept when applying the model
      inp2 = layers.Input(shape=(2,))
      out2 = model(inp2)
      self.assertIn('dropout', out2.op.name)

      # Test that argument is kept after loading a model
      config = model.get_config()
      model = training_lib.Model.from_config(config)
      self.assertIn('dropout', model.output.op.name)

  def test_node_construction(self):
    # test basics
    a = layers.Input(shape=(32,), name='input_a')
    b = layers.Input(shape=(32,), name='input_b')

    with self.assertRaises(ValueError):
      _ = layers.Input(shape=(32,), batch_shape=(10, 32))
    with self.assertRaises(ValueError):
      _ = layers.Input(shape=(32,), unknown_kwarg=None)

    self.assertListEqual(a.shape.as_list(), [None, 32])
    a_layer, a_node_index, a_tensor_index = a._keras_history
    b_layer, _, _ = b._keras_history
    self.assertEqual(len(a_layer._inbound_nodes), 1)
    self.assertEqual(a_tensor_index, 0)
    node = a_layer._inbound_nodes[a_node_index]
    self.assertEqual(node.outbound_layer, a_layer)

    self.assertListEqual(node.inbound_layers, [])
    self.assertListEqual(node.input_tensors, [a])
    self.assertListEqual(node.input_shapes, [(None, 32)])
    self.assertListEqual(node.output_tensors, [a])
    self.assertListEqual(node.output_shapes, [(None, 32)])

    dense = layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    self.assertEqual(len(dense._inbound_nodes), 2)
    self.assertEqual(len(dense._outbound_nodes), 0)
    self.assertEqual(dense._inbound_nodes[0].inbound_layers, a_layer)
    self.assertEqual(dense._inbound_nodes[0].outbound_layer, dense)
    self.assertEqual(dense._inbound_nodes[1].inbound_layers, b_layer)
    self.assertEqual(dense._inbound_nodes[1].outbound_layer, dense)
    self.assertIs(dense._inbound_nodes[0].input_tensors, a)
    self.assertIs(dense._inbound_nodes[1].input_tensors, b)

    # test layer properties
    test_layer = layers.Dense(16, name='test_layer')
    a_test = test_layer(a)
    self.assertListEqual(test_layer.kernel.shape.as_list(), [32, 16])
    self.assertIs(test_layer.input, a)
    self.assertIs(test_layer.output, a_test)
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, (None, 16))

    self.assertIs(dense.get_input_at(0), a)
    self.assertIs(dense.get_input_at(1), b)
    self.assertIs(dense.get_output_at(0), a_2)
    self.assertIs(dense.get_output_at(1), b_2)
    self.assertEqual(dense.get_input_shape_at(0), (None, 32))
    self.assertEqual(dense.get_input_shape_at(1), (None, 32))
    self.assertEqual(dense.get_output_shape_at(0), (None, 16))
    self.assertEqual(dense.get_output_shape_at(1), (None, 16))
    self.assertEqual(dense.get_input_mask_at(0), None)
    self.assertEqual(dense.get_input_mask_at(1), None)
    self.assertEqual(dense.get_output_mask_at(0), None)
    self.assertEqual(dense.get_output_mask_at(1), None)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_multi_input_layer(self):
    with self.cached_session():
      # test multi-input layer
      a = layers.Input(shape=(32,), name='input_a')
      b = layers.Input(shape=(32,), name='input_b')

      dense = layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)

      merged = layers.concatenate([a_2, b_2], name='merge')
      self.assertListEqual(merged.shape.as_list(), [None, 16 * 2])
      merge_layer, merge_node_index, merge_tensor_index = merged._keras_history

      self.assertEqual(merge_node_index, 0)
      self.assertEqual(merge_tensor_index, 0)

      self.assertEqual(len(merge_layer._inbound_nodes), 1)
      self.assertEqual(len(merge_layer._outbound_nodes), 0)

      self.assertEqual(len(merge_layer._inbound_nodes[0].input_tensors), 2)
      self.assertEqual(len(merge_layer._inbound_nodes[0].inbound_layers), 2)

      c = layers.Dense(64, name='dense_2')(merged)
      d = layers.Dense(5, name='dense_3')(c)

      model = training_lib.Model(inputs=[a, b], outputs=[c, d], name='model')
      self.assertEqual(len(model.layers), 6)
      output_shapes = model.compute_output_shape([(None, 32), (None, 32)])
      self.assertListEqual(output_shapes[0].as_list(), [None, 64])
      self.assertListEqual(output_shapes[1].as_list(), [None, 5])
      self.assertListEqual(
          model.compute_mask([a, b], [None, None]), [None, None])

      # we don't check names of first 2 layers (inputs) because
      # ordering of same-level layers is not fixed
      self.assertListEqual([l.name for l in model.layers][2:],
                           ['dense_1', 'merge', 'dense_2', 'dense_3'])
      self.assertListEqual([l.name for l in model._input_layers],
                           ['input_a', 'input_b'])
      self.assertListEqual([l.name for l in model._output_layers],
                           ['dense_2', 'dense_3'])

      # actually run model
      fn = backend.function(model.inputs, model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 64), (10, 5)])

      # test get_source_inputs
      self._assertAllIs(layer_utils.get_source_inputs(c), [a, b])

      # serialization / deserialization
      json_config = model.to_json()
      recreated_model = models.model_from_json(json_config)
      recreated_model.compile('rmsprop', 'mse')

      self.assertListEqual([l.name for l in recreated_model.layers][2:],
                           ['dense_1', 'merge', 'dense_2', 'dense_3'])
      self.assertListEqual([l.name for l in recreated_model._input_layers],
                           ['input_a', 'input_b'])
      self.assertListEqual([l.name for l in recreated_model._output_layers],
                           ['dense_2', 'dense_3'])

      fn = backend.function(recreated_model.inputs, recreated_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 64), (10, 5)])

  def test_multi_output_layer_output_names(self):
    inp = layers.Input(name='inp', shape=(None,), dtype=dtypes.float32)

    class _MultiOutput(layers.Layer):

      def call(self, x):
        return x + 1., x + 2.

    out = _MultiOutput(name='out')(inp)
    model = training_lib.Model(inp, out)
    self.assertEqual(['out', 'out_1'], model.output_names)
    self.assertAllClose([2., 3.], model(1.))

  def test_recursion(self):
    with ops.Graph().as_default(), self.cached_session():
      a = layers.Input(shape=(32,), name='input_a')
      b = layers.Input(shape=(32,), name='input_b')

      dense = layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = layers.concatenate([a_2, b_2], name='merge')
      c = layers.Dense(64, name='dense_2')(merged)
      d = layers.Dense(5, name='dense_3')(c)

      model = training_lib.Model(inputs=[a, b], outputs=[c, d], name='model')

      e = layers.Input(shape=(32,), name='input_e')
      f = layers.Input(shape=(32,), name='input_f')
      self.assertEqual(len(model.inputs), 2)
      g, h = model([e, f])
      self.assertEqual(len(model.inputs), 2)
      self.assertEqual(g.name, 'model/dense_2/BiasAdd:0')

      self.assertListEqual(g.shape.as_list(), c.shape.as_list())
      self.assertListEqual(h.shape.as_list(), d.shape.as_list())

      # test separate manipulation of different layer outputs
      i = layers.Dense(7, name='dense_4')(h)

      final_model = training_lib.Model(
          inputs=[e, f], outputs=[i, g], name='final')
      self.assertEqual(len(final_model.inputs), 2)
      self.assertEqual(len(final_model.outputs), 2)
      self.assertEqual(len(final_model.layers), 4)

      # we don't check names of first 2 layers (inputs) because
      # ordering of same-level layers is not fixed
      self.assertListEqual([layer.name for layer in final_model.layers][2:],
                           ['model', 'dense_4'])
      self.assertListEqual(
          model.compute_mask([e, f], [None, None]), [None, None])
      self.assertListEqual(
          final_model.compute_output_shape([(10, 32), (10, 32)]), [(10, 7),
                                                                   (10, 64)])

      # run recursive model
      fn = backend.function(final_model.inputs, final_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 7), (10, 64)])

      # test serialization
      model_config = final_model.get_config()
      recreated_model = models.Model.from_config(model_config)

      fn = backend.function(recreated_model.inputs, recreated_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 7), (10, 64)])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_multi_input_multi_output_recursion(self):
    with self.cached_session():
      # test multi-input multi-output
      a = layers.Input(shape=(32,), name='input_a')
      b = layers.Input(shape=(32,), name='input_b')

      dense = layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = layers.concatenate([a_2, b_2], name='merge')
      c = layers.Dense(64, name='dense_2')(merged)
      d = layers.Dense(5, name='dense_3')(c)

      model = training_lib.Model(inputs=[a, b], outputs=[c, d], name='model')

      j = layers.Input(shape=(32,), name='input_j')
      k = layers.Input(shape=(32,), name='input_k')
      _, n = model([j, k])

      o = layers.Input(shape=(32,), name='input_o')
      p = layers.Input(shape=(32,), name='input_p')
      q, _ = model([o, p])

      self.assertListEqual(n.shape.as_list(), [None, 5])
      self.assertListEqual(q.shape.as_list(), [None, 64])
      s = layers.concatenate([n, q], name='merge_nq')
      self.assertListEqual(s.shape.as_list(), [None, 64 + 5])

      # test with single output as 1-elem list
      multi_io_model = training_lib.Model([j, k, o, p], [s])

      fn = backend.function(multi_io_model.inputs, multi_io_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      # test with single output as tensor
      multi_io_model = training_lib.Model([j, k, o, p], s)

      fn = backend.function(multi_io_model.inputs, multi_io_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      # note that the output of the function will still be a 1-elem list
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      # test serialization
      model_config = multi_io_model.get_config()
      recreated_model = models.Model.from_config(model_config)

      fn = backend.function(recreated_model.inputs, recreated_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      # note that the output of the function will still be a 1-elem list
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      config = model.get_config()
      models.Model.from_config(config)

      model.summary()
      json_str = model.to_json()
      models.model_from_json(json_str)

      if yaml is not None:
        yaml_str = model.to_yaml()
        models.model_from_yaml(yaml_str)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_invalid_graphs(self):
    a = layers.Input(shape=(32,), name='input_a')
    b = layers.Input(shape=(32,), name='input_b')

    dense = layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)
    merged = layers.concatenate([a_2, b_2], name='merge')
    c = layers.Dense(64, name='dense_2')(merged)
    d = layers.Dense(5, name='dense_3')(c)

    model = training_lib.Model(inputs=[a, b], outputs=[c, d], name='model')

    # input is not an Input tensor
    j = layers.Input(shape=(32,), name='input_j')
    j = layers.Dense(32)(j)
    k = layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    with self.assertRaises(Exception):
      training_lib.Model([j, k], [m, n])

    # disconnected graph
    j = layers.Input(shape=(32,), name='input_j')
    k = layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      training_lib.Model([j], [m, n])

    # redundant outputs
    j = layers.Input(shape=(32,), name='input_j')
    k = layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    training_lib.Model([j, k], [m, n, n])

    # redundant inputs
    j = layers.Input(shape=(32,), name='input_j')
    k = layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      training_lib.Model([j, k, j], [m, n])

    # i have not idea what I'm doing: garbage as inputs/outputs
    j = layers.Input(shape=(32,), name='input_j')
    k = layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      training_lib.Model([j, k], [m, n, 0])

  def test_raw_tf_compatibility(self):
    with ops.Graph().as_default():
      # test calling layers/models on TF tensors
      a = layers.Input(shape=(32,), name='input_a')
      b = layers.Input(shape=(32,), name='input_b')

      dense = layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = layers.concatenate([a_2, b_2], name='merge')
      c = layers.Dense(64, name='dense_2')(merged)
      d = layers.Dense(5, name='dense_3')(c)

      model = training_lib.Model(inputs=[a, b], outputs=[c, d], name='model')

      j = layers.Input(shape=(32,), name='input_j')
      k = layers.Input(shape=(32,), name='input_k')
      self.assertEqual(len(model.inputs), 2)
      m, n = model([j, k])
      self.assertEqual(len(model.inputs), 2)
      tf_model = training_lib.Model([j, k], [m, n])

      j_tf = array_ops.placeholder(dtype=dtypes.float32, shape=(None, 32))
      k_tf = array_ops.placeholder(dtype=dtypes.float32, shape=(None, 32))
      m_tf, n_tf = tf_model([j_tf, k_tf])
      self.assertListEqual(m_tf.shape.as_list(), [None, 64])
      self.assertListEqual(n_tf.shape.as_list(), [None, 5])

      # test merge
      layers.concatenate([j_tf, k_tf], axis=1)
      layers.add([j_tf, k_tf])

      # test tensor input
      x = array_ops.placeholder(shape=(None, 2), dtype=dtypes.float32)
      layers.InputLayer(input_tensor=x)

      x = layers.Input(tensor=x)
      layers.Dense(2)(x)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_basic_masking(self):
    a = layers.Input(shape=(10, 32), name='input_a')
    b = layers.Masking()(a)
    model = training_lib.Model(a, b)
    self.assertEqual(model.output_mask.shape.as_list(), [None, 10])

  def testMaskingSingleInput(self):

    class MaskedLayer(layers.Layer):

      def call(self, inputs, mask=None):
        if mask is not None:
          return inputs * mask
        return inputs

      def compute_mask(self, inputs, mask=None):
        return array_ops.ones_like(inputs)

    if context.executing_eagerly():
      a = constant_op.constant([2] * 32)
      mask = constant_op.constant([0, 1] * 16)
      a._keras_mask = mask
      b = MaskedLayer().apply(a)
      self.assertTrue(hasattr(b, '_keras_mask'))
      self.assertAllEqual(
          self.evaluate(array_ops.ones_like(mask)),
          self.evaluate(getattr(b, '_keras_mask')))
      self.assertAllEqual(self.evaluate(a * mask), self.evaluate(b))
    else:
      x = input_layer_lib.Input(shape=(32,))
      y = MaskedLayer()(x)  # pylint: disable=not-callable
      network = functional.Functional(x, y)

      # test callability on Input
      x_2 = input_layer_lib.Input(shape=(32,))
      y_2 = network(x_2)
      self.assertEqual(y_2.shape.as_list(), [None, 32])

      # test callability on regular tensor
      x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
      y_2 = network(x_2)
      self.assertEqual(y_2.shape.as_list(), [None, 32])

  def test_activity_regularization_with_model_composition(self):

    def reg(x):
      return math_ops.reduce_sum(x)

    net_a_input = input_layer_lib.Input((2,))
    net_a = net_a_input
    net_a = layers.Dense(
        2, kernel_initializer='ones', use_bias=False, activity_regularizer=reg)(
            net_a)
    model_a = training_lib.Model([net_a_input], [net_a])

    net_b_input = input_layer_lib.Input((2,))
    net_b = model_a(net_b_input)
    model_b = training_lib.Model([net_b_input], [net_b])

    model_b.compile(optimizer='sgd', loss=None)
    x = np.ones((1, 2))
    loss = model_b.evaluate(x)
    self.assertEqual(loss, 4.)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_layer_sharing_at_heterogenous_depth(self):
    x_val = np.random.random((10, 5))

    x = input_layer_lib.Input(shape=(5,))
    a = layers.Dense(5, name='A')
    b = layers.Dense(5, name='B')
    output = a(b(a(b(x))))
    m = training_lib.Model(x, output)
    m.run_eagerly = testing_utils.should_run_eagerly()

    output_val = m.predict(x_val)

    config = m.get_config()
    weights = m.get_weights()

    m2 = models.Model.from_config(config)
    m2.set_weights(weights)

    output_val_2 = m2.predict(x_val)
    self.assertAllClose(output_val, output_val_2, atol=1e-6)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_layer_sharing_at_heterogenous_depth_with_concat(self):
    input_shape = (16, 9, 3)
    input_layer = input_layer_lib.Input(shape=input_shape)

    a = layers.Dense(3, name='dense_A')
    b = layers.Dense(3, name='dense_B')
    c = layers.Dense(3, name='dense_C')

    x1 = b(a(input_layer))
    x2 = a(c(input_layer))
    output = layers.concatenate([x1, x2])

    m = training_lib.Model(inputs=input_layer, outputs=output)
    m.run_eagerly = testing_utils.should_run_eagerly()

    x_val = np.random.random((10, 16, 9, 3))
    output_val = m.predict(x_val)

    config = m.get_config()
    weights = m.get_weights()

    m2 = models.Model.from_config(config)
    m2.set_weights(weights)

    output_val_2 = m2.predict(x_val)
    self.assertAllClose(output_val, output_val_2, atol=1e-6)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_explicit_training_argument(self):
    a = layers.Input(shape=(2,))
    b = layers.Dropout(0.5)(a)
    base_model = training_lib.Model(a, b)

    a = layers.Input(shape=(2,))
    b = base_model(a, training=False)
    model = training_lib.Model(a, b)

    x = np.ones((100, 2))
    y = np.ones((100, 2))
    model.compile(
        optimizer='sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    loss = model.train_on_batch(x, y)
    self.assertEqual(loss, 0)  # In inference mode, output is equal to input.

    a = layers.Input(shape=(2,))
    b = base_model(a, training=True)
    model = training_lib.Model(a, b)
    preds = model.predict(x)
    self.assertEqual(np.min(preds), 0.)  # At least one unit was dropped.

  @combinations.generate(combinations.keras_mode_combinations())
  def test_mask_derived_from_keras_layer(self):
    inputs = input_layer_lib.Input((5, 10))
    mask = input_layer_lib.Input((5,))
    outputs = layers.RNN(layers.LSTMCell(100))(inputs, mask=mask)
    model = training_lib.Model([inputs, mask], outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[np.ones((10, 5, 10)), np.zeros((10, 5))],
        y=np.zeros((10, 100)),
        batch_size=2)
    # All data is masked, returned values are 0's.
    self.assertEqual(history.history['loss'][0], 0.0)
    history = model.fit(
        x=[np.ones((10, 5, 10)), np.ones((10, 5))],
        y=np.zeros((10, 100)),
        batch_size=2)
    # Data is not masked, returned values are random.
    self.assertGreater(history.history['loss'][0], 0.0)

    model = training_lib.Model.from_config(model.get_config())
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[np.ones((10, 5, 10)), np.zeros((10, 5))],
        y=np.zeros((10, 100)),
        batch_size=2)
    # All data is masked, returned values are 0's.
    self.assertEqual(history.history['loss'][0], 0.0)
    history = model.fit(
        x=[np.ones((10, 5, 10)), np.ones((10, 5))],
        y=np.zeros((10, 100)),
        batch_size=2)
    # Data is not masked, returned values are random.
    self.assertGreater(history.history['loss'][0], 0.0)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_call_arg_derived_from_keras_layer(self):

    class MyAdd(layers.Layer):

      def call(self, x1, x2):
        return x1 + x2

    input1 = input_layer_lib.Input(10)
    input2 = input_layer_lib.Input(10)
    outputs = MyAdd()(input1, input2)
    model = training_lib.Model([input1, input2], outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
        y=10 * np.ones((10, 10)),
        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

    # Check serialization.
    model = training_lib.Model.from_config(
        model.get_config(), custom_objects={'MyAdd': MyAdd})
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
        y=10 * np.ones((10, 10)),
        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_call_kwarg_derived_from_keras_layer(self):

    class MaybeAdd(layers.Layer):

      def call(self, x1, x2=None):
        if x2 is not None:
          return x1 + x2
        return x1

    input1 = input_layer_lib.Input(10)
    input2 = input_layer_lib.Input(10)
    outputs = MaybeAdd()(input1, x2=input2)
    model = training_lib.Model([input1, input2], outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
        y=10 * np.ones((10, 10)),
        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

    model = training_lib.Model.from_config(
        model.get_config(), custom_objects={'MaybeAdd': MaybeAdd})
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
        y=10 * np.ones((10, 10)),
        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_composite_call_kwarg_derived_from_keras_layer(self):

    # Create a test layer that accepts composite tensor inputs.
    class MaybeAdd(layers.Layer):

      def call(self, x1, x2=None):
        # We need to convert this to a tensor for loss calculations -
        # losses don't play nicely with ragged tensors yet.
        if x2 is not None:
          return (x1 + x2).to_tensor(default_value=0)
        return x1.to_tensor(default_value=0)

    input1 = input_layer_lib.Input((None,), ragged=True)
    input2 = input_layer_lib.Input((None,), ragged=True)
    outputs = MaybeAdd()(input1, x2=input2)
    model = training_lib.Model([input1, input2], outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    input_data = [
        ragged_factory_ops.constant([[3.0, 3.0], [3.0, 3.0], [3.0]]),
        ragged_factory_ops.constant([[7.0, 7.0], [7.0, 7.0], [7.0]])
    ]
    expected_data = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 0.0]])

    history = model.fit(x=input_data, y=expected_data)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

    model = training_lib.Model.from_config(
        model.get_config(), custom_objects={'MaybeAdd': MaybeAdd})
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(x=input_data, y=expected_data)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_call_nested_arg_derived_from_keras_layer(self):

    class AddAll(layers.Layer):

      def call(self, x1, x2, x3=None):
        out = x1 + x2
        if x3 is not None:
          for t in x3.values():
            out += t
        return out

    input1 = input_layer_lib.Input(10)
    input2 = input_layer_lib.Input(10)
    input3 = input_layer_lib.Input(10)
    outputs = AddAll()(
        input1,
        4 * array_ops.ones((1, 10)),
        x3={
            'a': input2,
            'b': input3,
            'c': 5 * array_ops.ones((1, 10))
        })
    model = training_lib.Model([input1, input2, input3], outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[np.ones((10, 10)), 2 * np.ones((10, 10)), 3 * np.ones((10, 10))],
        y=15 * np.ones((10, 10)),
        batch_size=2)
    # Check that all inputs were correctly added.
    self.assertEqual(history.history['loss'][0], 0.0)

    model = training_lib.Model.from_config(
        model.get_config(), custom_objects={'AddAll': AddAll})
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(
        x=[np.ones((10, 10)), 2 * np.ones((10, 10)), 3 * np.ones((10, 10))],
        y=15 * np.ones((10, 10)),
        batch_size=2)
    # Check that all inputs were correctly added.
    self.assertEqual(history.history['loss'][0], 0.0)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_multi_output_model_with_none_masking(self):
    def func(x):
      return [x * 0.2, x * 0.3]

    def output_shape(input_shape):
      return [input_shape, input_shape]

    i = layers.Input(shape=(3, 2, 1))
    o = layers.Lambda(function=func, output_shape=output_shape)(i)

    self.assertEqual(backend.int_shape(o[0]), (None, 3, 2, 1))
    self.assertEqual(backend.int_shape(o[1]), (None, 3, 2, 1))

    o = layers.add(o)
    model = training_lib.Model(i, o)
    model.run_eagerly = testing_utils.should_run_eagerly()

    i2 = layers.Input(shape=(3, 2, 1))
    o2 = model(i2)
    model2 = training_lib.Model(i2, o2)
    model2.run_eagerly = testing_utils.should_run_eagerly()

    x = np.random.random((4, 3, 2, 1))
    out = model2.predict(x)
    assert out.shape == (4, 3, 2, 1)
    self.assertAllClose(out, x * 0.2 + x * 0.3, atol=1e-4)

  @combinations.generate(combinations.keras_mode_combinations())
  def test_constant_initializer_with_numpy(self):
    initializer = initializers.Constant(np.ones((3, 2)))
    model = sequential.Sequential()
    model.add(layers.Dense(2, input_shape=(3,), kernel_initializer=initializer))
    model.add(layers.Dense(3))
    model.compile(
        loss='mse',
        optimizer='sgd',
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())

    json_str = model.to_json()
    models.model_from_json(json_str)

    if yaml is not None:
      yaml_str = model.to_yaml()
      models.model_from_yaml(yaml_str)

  def test_subclassed_error_if_init_not_called(self):

    class MyNetwork(training_lib.Model):

      def __init__(self):
        self._foo = [layers.Dense(10), layers.Dense(10)]

    with self.assertRaisesRegexp(RuntimeError, 'forgot to call'):
      MyNetwork()

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_int_input_shape(self):
    inputs = input_layer_lib.Input(10)
    self.assertEqual([None, 10], inputs.shape.as_list())

    inputs_with_batch = input_layer_lib.Input(batch_size=20, shape=5)
    self.assertEqual([20, 5], inputs_with_batch.shape.as_list())

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_model_initialization(self):
    # Functional model
    inputs = input_layer_lib.Input(shape=(32,))
    outputs = layers.Dense(4)(inputs)

    with self.assertRaisesRegexp(TypeError,
                                 'got an unexpected keyword argument'):
      model = training_lib.Model(
          inputs, outputs, name='m', trainable=False, dtype='int64')
    with self.assertRaisesRegexp(TypeError,
                                 'got an unexpected keyword argument'):
      model = training_lib.Model(
          inputs, outputs, name='m', trainable=False, dynamic=False)

    model = training_lib.Model(inputs, outputs, name='m', trainable=False)
    self.assertEqual('m', model.name)
    self.assertFalse(model.trainable)
    self.assertFalse(model.dynamic)

    class SubclassModel(training_lib.Model):
      pass
    # Subclassed model
    model = SubclassModel(
        name='subclassed', trainable=True, dtype='int64', dynamic=True)
    self.assertEqual('subclassed', model.name)
    self.assertTrue(model.dynamic)
    self.assertTrue(model.trainable)
    w = model.add_weight('w', [], initializer=initializers.Constant(1))
    self.assertEqual(dtypes.int64, w.dtype)

  def test_disconnected_inputs(self):
    input_tensor1 = input_layer_lib.Input(shape=[200], name='a')
    input_tensor2 = input_layer_lib.Input(shape=[10], name='b')
    output_tensor1 = layers.Dense(units=10)(input_tensor1)

    net = functional.Functional(
        inputs=[input_tensor1, input_tensor2], outputs=[output_tensor1])
    net2 = functional.Functional.from_config(net.get_config())
    self.assertLen(net2.inputs, 2)
    self.assertEqual('a', net2.layers[0].name)
    self.assertEqual('b', net2.layers[1].name)

  @combinations.generate(combinations.keras_model_type_combinations())
  def test_dependency_tracking(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.trackable = Checkpoint()
    self.assertIn('trackable', model._unconditional_dependency_names)
    self.assertEqual(model.trackable, model._lookup_dependency('trackable'))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_model_construction_in_tf_function(self):

    d = {'model': None}

    @def_function.function
    def fn(x):
      if d['model'] is None:
        # Check that Functional can be built in a `tf.function`.
        inputs = input_layer_lib.Input(10)
        outputs = layers.Dense(1)(inputs)
        model = functional.Functional(inputs, outputs)
        d['model'] = model
      else:
        model = d['model']

      return model(x)

    x = array_ops.ones((10, 10))
    y = fn(x)
    self.assertEqual(y.shape.as_list(), [10, 1])


class DeferredModeTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testSimpleNetworkBuilding(self):
    inputs = input_layer_lib.Input(shape=(32,))
    if context.executing_eagerly():
      self.assertEqual(inputs.dtype.name, 'float32')
      self.assertEqual(inputs.shape.as_list(), [None, 32])

    x = layers.Dense(2)(inputs)
    if context.executing_eagerly():
      self.assertEqual(x.dtype.name, 'float32')
      self.assertEqual(x.shape.as_list(), [None, 2])

    outputs = layers.Dense(4)(x)
    network = functional.Functional(inputs, outputs)
    self.assertIsInstance(network, functional.Functional)

    if context.executing_eagerly():
      # It should be possible to call such a network on EagerTensors.
      inputs = constant_op.constant(
          np.random.random((10, 32)).astype('float32'))
      outputs = network(inputs)
      self.assertEqual(outputs.shape.as_list(), [10, 4])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testMultiIONetworkBuilding(self):
    input_a = input_layer_lib.Input(shape=(32,))
    input_b = input_layer_lib.Input(shape=(16,))
    a = layers.Dense(16)(input_a)

    class AddLayer(layers.Layer):

      def call(self, inputs):
        return inputs[0] + inputs[1]

    c = AddLayer()([a, input_b])  # pylint: disable=not-callable
    c = layers.Dense(2)(c)

    network = functional.Functional([input_a, input_b], [a, c])
    if context.executing_eagerly():
      a_val = constant_op.constant(
          np.random.random((10, 32)).astype('float32'))
      b_val = constant_op.constant(
          np.random.random((10, 16)).astype('float32'))
      outputs = network([a_val, b_val])
      self.assertEqual(len(outputs), 2)
      self.assertEqual(outputs[0].shape.as_list(), [10, 16])
      self.assertEqual(outputs[1].shape.as_list(), [10, 2])


class DefaultShapeInferenceBehaviorTest(keras_parameterized.TestCase):

  def _testShapeInference(self, model, input_shape, expected_output_shape):
    input_value = np.random.random(input_shape)
    output_value = model.predict(input_value)
    self.assertEqual(output_value.shape, expected_output_shape)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testSingleInputCase(self):

    class LayerWithOneInput(layers.Layer):

      def build(self, input_shape):
        self.w = array_ops.ones(shape=(3, 4))

      def call(self, inputs):
        return backend.dot(inputs, self.w)

    inputs = input_layer_lib.Input(shape=(3,))
    layer = LayerWithOneInput()

    if context.executing_eagerly():
      self.assertEqual(
          layer.compute_output_shape((None, 3)).as_list(), [None, 4])
      # As a side-effect, compute_output_shape builds the layer.
      self.assertTrue(layer.built)
      # We can still query the layer's compute_output_shape with compatible
      # input shapes.
      self.assertEqual(
          layer.compute_output_shape((6, 3)).as_list(), [6, 4])

    outputs = layer(inputs)
    model = training_lib.Model(inputs, outputs)
    self._testShapeInference(model, (2, 3), (2, 4))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testMultiInputOutputCase(self):

    class MultiInputOutputLayer(layers.Layer):

      def build(self, input_shape):
        self.w = array_ops.ones(shape=(3, 4))

      def call(self, inputs):
        a = backend.dot(inputs[0], self.w)
        b = a + inputs[1]
        return [a, b]

    input_a = input_layer_lib.Input(shape=(3,))
    input_b = input_layer_lib.Input(shape=(4,))
    output_a, output_b = MultiInputOutputLayer()([input_a, input_b])
    model = training_lib.Model([input_a, input_b], [output_a, output_b])
    output_a_val, output_b_val = model.predict(
        [np.random.random((2, 3)), np.random.random((2, 4))])
    self.assertEqual(output_a_val.shape, (2, 4))
    self.assertEqual(output_b_val.shape, (2, 4))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTrainingArgument(self):

    class LayerWithTrainingArg(layers.Layer):

      def build(self, input_shape):
        self.w = array_ops.ones(shape=(3, 4))

      def call(self, inputs, training):
        return backend.dot(inputs, self.w)

    inputs = input_layer_lib.Input(shape=(3,))
    outputs = LayerWithTrainingArg()(inputs, training=False)
    model = training_lib.Model(inputs, outputs)
    self._testShapeInference(model, (2, 3), (2, 4))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoneInShape(self):

    class Model(training_lib.Model):

      def __init__(self):
        super(Model, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(3)

      def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

    model = Model()
    model.build(tensor_shape.TensorShape((None, None, None, 1)))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(model.weights,
                    'Model should have its weights created as it '
                    'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)
    self.assertEqual(output.shape, (1, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoneInShapeWithCompoundModel(self):

    class BasicBlock(training_lib.Model):

      def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)
        self.pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(3)

      def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense(x)
        return x

    class CompoundModel(training_lib.Model):

      def __init__(self):
        super(CompoundModel, self).__init__()
        self.block = BasicBlock()

      def call(self, x):
        x = self.block(x)  # pylint: disable=not-callable
        return x

    model = CompoundModel()
    model.build(tensor_shape.TensorShape((None, None, None, 1)))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(model.weights,
                    'Model should have its weights created as it '
                    'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)  # pylint: disable=not-callable
    self.assertEqual(output.shape, (1, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoneInShapeWithFunctionalAPI(self):

    class BasicBlock(training_lib.Model):
      # Inheriting from layers.Layer since we are calling this layer
      # inside a model created using functional API.

      def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)

      def call(self, x):
        x = self.conv1(x)
        return x

    input_layer = layers.Input(shape=(None, None, 1))
    x = BasicBlock()(input_layer)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Dense(3)(x)

    model = training_lib.Model(inputs=input_layer, outputs=output_layer)

    model.build(tensor_shape.TensorShape((None, None, None, 1)))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(model.weights,
                    'Model should have its weights created as it '
                    'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)
    self.assertEqual(output.shape, (1, 3))

  @combinations.generate(combinations.keras_mode_combinations())
  def test_sequential_as_downstream_of_masking_layer(self):
    inputs = layers.Input(shape=(3, 4))
    x = layers.Masking(mask_value=0., input_shape=(3, 4))(inputs)

    s = sequential.Sequential()
    s.add(layers.Dense(5, input_shape=(4,)))

    x = layers.wrappers.TimeDistributed(s)(x)
    model = training_lib.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())

    model_input = np.random.randint(
        low=1, high=5, size=(10, 3, 4)).astype('float32')
    for i in range(4):
      model_input[i, i:, :] = 0.
    model.fit(model_input,
              np.random.random((10, 3, 5)), epochs=1, batch_size=6)

    if not context.executing_eagerly():
      # Note: this doesn't work in eager due to DeferredTensor/ops compatibility
      # issue.
      mask_outputs = [model.layers[1].compute_mask(model.layers[1].input)]
      mask_outputs += [model.layers[2].compute_mask(
          model.layers[2].input, mask_outputs[-1])]
      func = backend.function([model.input], mask_outputs)
      mask_outputs_val = func([model_input])
      self.assertAllClose(mask_outputs_val[0], np.any(model_input, axis=-1))
      self.assertAllClose(mask_outputs_val[1], np.any(model_input, axis=-1))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_external_keras_serialization_compat_input_layers(self):
    inputs = input_layer_lib.Input(shape=(10,))
    outputs = layers.Dense(1)(inputs)
    model = training_lib.Model(inputs, outputs)
    config = model.get_config()
    # Checks that single inputs and outputs are still saved as 1-element lists.
    # Saving as 1-element lists or not is equivalent in TF Keras, but only the
    # 1-element list format is supported in TF.js and keras-team/Keras.
    self.assertLen(config['input_layers'], 1)
    self.assertLen(config['output_layers'], 1)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_external_keras_serialization_compat_inbound_nodes(self):
    # Check single Tensor input.
    inputs = input_layer_lib.Input(shape=(10,), name='in')
    outputs = layers.Dense(1)(inputs)
    model = training_lib.Model(inputs, outputs)
    config = model.get_config()
    self.assertEqual(config['layers'][1]['inbound_nodes'], [[['in', 0, 0, {}]]])

    # Check multiple Tensor input.
    inputs1 = input_layer_lib.Input(shape=(10,), name='in1')
    inputs2 = input_layer_lib.Input(shape=(10,), name='in2')
    outputs = layers.Add()([inputs1, inputs2])
    model = training_lib.Model([inputs1, inputs2], outputs)
    config = model.get_config()
    self.assertEqual(config['layers'][2]['inbound_nodes'],
                     [[['in1', 0, 0, {}], ['in2', 0, 0, {}]]])


class GraphUtilsTest(test.TestCase):

  def testGetReachableFromInputs(self):

    with ops.Graph().as_default(), self.cached_session():
      pl_1 = array_ops.placeholder(shape=None, dtype='float32')
      pl_2 = array_ops.placeholder(shape=None, dtype='float32')
      pl_3 = array_ops.placeholder(shape=None, dtype='float32')
      x_1 = pl_1 + pl_2
      x_2 = pl_2 * 2
      x_3 = pl_3 + 1
      x_4 = x_1 + x_2
      x_5 = x_3 * pl_1

      self.assertEqual(
          tf_utils.get_reachable_from_inputs([pl_1]),
          {pl_1, x_1, x_4, x_5, x_1.op, x_4.op, x_5.op})
      self.assertEqual(
          tf_utils.get_reachable_from_inputs([pl_1, pl_2]),
          {pl_1, pl_2, x_1, x_2, x_4, x_5, x_1.op, x_2.op, x_4.op, x_5.op})
      self.assertEqual(
          tf_utils.get_reachable_from_inputs([pl_3]),
          {pl_3, x_3, x_5, x_3.op, x_5.op})
      self.assertEqual(
          tf_utils.get_reachable_from_inputs([x_3]), {x_3, x_5, x_5.op})


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class NestedNetworkTest(keras_parameterized.TestCase):

  def test_nested_inputs_network(self):
    inputs = {
        'x1': input_layer_lib.Input(shape=(1,)),
        'x2': input_layer_lib.Input(shape=(1,))
    }
    outputs = layers.Add()([inputs['x1'], inputs['x2']])
    network = functional.Functional(inputs, outputs)

    network = functional.Functional.from_config(network.get_config())

    result_tensor = network({
        'x': array_ops.ones((1, 1), 'float32'),
        'y': array_ops.ones((1, 1), 'float32')
    })
    result = self.evaluate(result_tensor)
    self.assertAllEqual(result, [[2.]])

    # TODO(b/122726584): Investigate why concrete batch is flaky in some builds.
    output_shape = network.compute_output_shape({
        'x1': (None, 1),
        'x2': (None, 1)
    })
    self.assertListEqual(output_shape.as_list(), [None, 1])

  def test_nested_outputs_network(self):
    inputs = input_layer_lib.Input(shape=(1,))
    outputs = {
        'x+x': layers.Add()([inputs, inputs]),
        'x*x': layers.Multiply()([inputs, inputs])
    }

    network = functional.Functional(inputs, outputs)

    network = functional.Functional.from_config(network.get_config())

    result_tensor = network(array_ops.ones((1, 1), 'float32'))
    result = self.evaluate(result_tensor)
    self.assertAllEqual(result['x+x'], [[2.]])
    self.assertAllEqual(result['x*x'], [[1.]])

    output_shape = network.compute_output_shape((None, 1))
    self.assertListEqual(output_shape['x+x'].as_list(), [None, 1])
    self.assertListEqual(output_shape['x*x'].as_list(), [None, 1])

  def test_nested_network_inside_network(self):
    inner_inputs = {
        'x1': input_layer_lib.Input(shape=(1,)),
        'x2': input_layer_lib.Input(shape=(1,))
    }
    inner_outputs = {
        'x1+x2': layers.Add()([inner_inputs['x1'], inner_inputs['x2']]),
        'x1*x2': layers.Multiply()([inner_inputs['x1'], inner_inputs['x2']])
    }
    inner_network = functional.Functional(
        inner_inputs, inner_outputs)

    inputs = [
        input_layer_lib.Input(shape=(1,)),
        input_layer_lib.Input(shape=(1,))
    ]
    middle = inner_network({'x1': inputs[0], 'x2': inputs[1]})
    outputs = layers.Add()([middle['x1+x2'], middle['x1*x2']])
    network = functional.Functional(inputs, outputs)

    network = functional.Functional.from_config(network.get_config())

    # Computes: `(x1+x2) + (x1*x2)`
    result_tensor = network(
        [array_ops.ones((1, 1), 'float32'),
         array_ops.ones((1, 1), 'float32')])
    result = self.evaluate(result_tensor)
    self.assertAllEqual(result, [[3.]])

    output_shape = network.compute_output_shape([(None, 1), (None, 1)])
    self.assertListEqual(output_shape.as_list(), [None, 1])

  def test_updates_with_direct_call(self):
    inputs = input_layer_lib.Input(shape=(10,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(10)(x)
    model = training_lib.Model(inputs, x)

    ph = backend.placeholder(shape=(10, 10))
    model(ph)

    self.assertLen(model.updates, 4)

  def test_dict_mapping_input(self):

    class ReturnFirst(layers.Layer):

      def call(self, inputs):
        b, _ = inputs
        return b

    # Checks that inputs are put in same order as the
    # Model was constructed with.
    b = input_layer_lib.Input(shape=(10,), name='b')
    a = input_layer_lib.Input(shape=(10,), name='a')
    outputs = ReturnFirst()([b, a])

    b_val = array_ops.ones((10, 10))
    a_val = array_ops.zeros((10, 10))

    model = training_lib.Model([b, a], outputs)
    res = model({'a': a_val, 'b': b_val})
    self.assertAllClose(self.evaluate(res), self.evaluate(b_val))

    reversed_model = training_lib.Model([a, b], outputs)
    res = reversed_model({'a': a_val, 'b': b_val})
    self.assertAllClose(self.evaluate(res), self.evaluate(b_val))

  def test_dict_mapping_single_input(self):
    b = input_layer_lib.Input(shape=(1,), name='b')
    outputs = b * 2
    model = training_lib.Model(b, outputs)

    b_val = array_ops.ones((1, 1))
    extra_val = array_ops.ones((1, 10))

    inputs = {'a': extra_val, 'b': b_val}
    res = model(inputs)

    # Check that 'b' was used and 'a' was ignored.
    self.assertEqual(res.shape.as_list(), [1, 1])


@combinations.generate(combinations.keras_mode_combinations())
class AddLossTest(keras_parameterized.TestCase):

  def test_add_loss_outside_call_only_loss(self):
    inputs = input_layer_lib.Input((10,))
    mid = layers.Dense(10)(inputs)
    outputs = layers.Dense(1)(mid)
    model = training_lib.Model(inputs, outputs)
    model.add_loss(math_ops.reduce_mean(outputs))
    self.assertLen(model.losses, 1)

    initial_weights = model.get_weights()

    x = np.ones((10, 10))
    model.compile(
        'sgd',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, batch_size=2, epochs=1)

    model2 = model.from_config(model.get_config())
    model2.compile(
        'sgd',
        run_eagerly=testing_utils.should_run_eagerly())
    model2.set_weights(initial_weights)
    model2.fit(x, batch_size=2, epochs=1)

    # The TFOpLayer and the AddLoss layer are serialized.
    self.assertLen(model2.layers, 5)
    self.assertAllClose(model.get_weights(), model2.get_weights())

  def test_add_loss_outside_call_multiple_losses(self):
    inputs = input_layer_lib.Input((10,))
    x1 = layers.Dense(10)(inputs)
    x2 = layers.Dense(10)(x1)
    outputs = layers.Dense(1)(x2)
    model = training_lib.Model(inputs, outputs)
    model.add_loss(math_ops.reduce_sum(x1 * x2))
    model.add_loss(math_ops.reduce_mean(outputs))
    self.assertLen(model.losses, 2)

    initial_weights = model.get_weights()

    x, y = np.ones((10, 10)), np.ones((10, 1))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=2, epochs=1)

    model2 = model.from_config(model.get_config())
    model2.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model2.set_weights(initial_weights)
    model2.fit(x, y, batch_size=2, epochs=1)

    self.assertAllClose(model.get_weights(), model2.get_weights())


@combinations.generate(combinations.keras_mode_combinations())
class WeightAccessTest(keras_parameterized.TestCase):

  def test_functional_model(self):
    inputs = input_layer_lib.Input((10,))
    x1 = layers.Dense(10)(inputs)
    x2 = layers.Dense(10)(x1)
    outputs = layers.Dense(1)(x2)
    model = training_lib.Model(inputs, outputs)

    self.assertEqual(len(model.weights), 6)

  def test_sequential_model_with_input_shape(self):
    x1 = layers.Dense(10, input_shape=(10,))
    x2 = layers.Dense(10)
    x3 = layers.Dense(1)
    model = sequential.Sequential([x1, x2, x3])

    self.assertEqual(len(model.weights), 6)

  def test_sequential_model_without_input_shape(self):
    x1 = layers.Dense(10)
    x2 = layers.Dense(10)
    x3 = layers.Dense(1)
    model = sequential.Sequential([x1, x2, x3])

    with self.assertRaisesRegexp(
        ValueError, 'Weights for model .* have not yet been created'):
      _ = model.weights

  def test_subclass_model_with_build_method(self):

    class SubclassModel(models.Model):

      def build(self, input_shape):
        self.w = self.add_weight(shape=input_shape[-1], initializer='ones')

      def call(self, inputs):
        return inputs * self.w

    model = SubclassModel()

    with self.assertRaisesRegexp(
        ValueError, 'Weights for model .* have not yet been created'):
      _ = model.weights

    model(input_layer_lib.Input((10,)))
    self.assertEqual(len(model.weights), 1)

  def test_subclass_model_without_build_method(self):

    class SubclassModel(models.Model):

      def __init__(self):
        super(SubclassModel, self).__init__()
        self.w = self.add_weight(shape=(), initializer='ones')

      def call(self, inputs):
        return inputs * self.w

    model = SubclassModel()
    self.assertEqual(len(model.weights), 1)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class DTypeTest(keras_parameterized.TestCase):

  @testing_utils.enable_v2_dtype_behavior
  def test_graph_network_dtype(self):
    inputs = input_layer_lib.Input((10,))
    outputs = layers.Dense(10)(inputs)
    network = functional.Functional(inputs, outputs)
    self.assertEqual(network.dtype, 'float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_subclassed_network_dtype(self):

    class IdentityNetwork(training_lib.Model):

      def call(self, inputs):
        return inputs

    network = IdentityNetwork()
    self.assertEqual(network.dtype, 'float32')
    self.assertEqual(network(array_ops.constant(1, 'float64')).dtype, 'float32')

    network = IdentityNetwork(dtype='float16')
    self.assertEqual(network.dtype, 'float16')
    self.assertEqual(network(array_ops.constant(1, 'float64')).dtype, 'float16')

    network = IdentityNetwork(autocast=False)
    self.assertEqual(network.dtype, 'float32')
    self.assertEqual(network(array_ops.constant(1, 'float64')).dtype, 'float64')


class AttrTrackingLayer(base_layer.Layer):
  """Count how many times `dynamic` and `stateful` are called.

  These counts are used to test that the attribute cache behaves as expected.
  """
  def __init__(self, *args, **kwargs):
    self.stateful_count = 0
    self.dynamic_count = 0
    super(AttrTrackingLayer, self).__init__(*args, **kwargs)

  @base_layer.Layer.stateful.getter
  def stateful(self):
    self.stateful_count += 1
    return super(AttrTrackingLayer, self).stateful

  @property
  def dynamic(self):
    self.dynamic_count += 1
    return super(AttrTrackingLayer, self).dynamic


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CacheCorrectnessTest(keras_parameterized.TestCase):

  def layer_and_network_test(self):
    # Top level layer
    network = functional.Functional()

    layer_0 = AttrTrackingLayer()

    sub_network = functional.Functional()
    layer_1 = AttrTrackingLayer(dynamic=True)
    layer_2 = AttrTrackingLayer()
    sub_network.sub_layers = [layer_1, layer_2]

    network.sub_layer = layer_0

    for _ in range(2):
      self.assertEqual(network.dynamic, False)
      self.assertEqual(network.stateful, False)

      # The second pass should be a cache hit.
      self.assertEqual(layer_0.dynamic_count, 1)
      self.assertEqual(layer_0.stateful_count, 1)

    # Mutations of the sub-layer should force recalculation of the network's
    # stateful attribute. (mutations bubble up.)
    layer_0.stateful = True
    self.assertEqual(network.stateful, True)
    self.assertEqual(layer_0.stateful_count, 2)

    layer_0.stateful = False
    self.assertEqual(network.stateful, False)
    self.assertEqual(layer_0.stateful_count, 3)

    # But changing stateful should not affect dynamic.
    self.assertEqual(network.dynamic, False)
    self.assertEqual(layer_0.dynamic_count, 1)

    network.sub_network = sub_network

    # Adding to the topology should invalidate the cache and reflect in the top
    # level network.
    self.assertEqual(network.dynamic, True)
    self.assertEqual(layer_0.dynamic_count, 2)
    self.assertEqual(layer_1.dynamic_count, 1)

    # Still dynamic, but we need to recompute.
    sub_network.sub_layers.pop()
    self.assertEqual(network.dynamic, True)
    self.assertEqual(layer_0.dynamic_count, 3)
    self.assertEqual(layer_1.dynamic_count, 2)

    # Now that we've removed the dynamic layer deep in the layer hierarchy, we
    # need to make sure that that bubbles up through all the levels.
    sub_network.sub_layers.pop()
    self.assertEqual(network.dynamic, False)
    self.assertEqual(layer_0.dynamic_count, 4)
    self.assertEqual(layer_1.dynamic_count, 2)

    # Now check with a tracked dict.
    sub_network.sub_layers = {
        "layer_1": layer_1,
        "layer_2": layer_2,
    }

    self.assertEqual(network.dynamic, True)
    self.assertEqual(layer_0.dynamic_count, 5)
    self.assertEqual(layer_1.dynamic_count, 3)

    # In-place assignment should still invalidate the cache.
    sub_network.sub_layers["layer_1"] = layer_1
    self.assertEqual(network.dynamic, True)
    self.assertEqual(layer_0.dynamic_count, 6)
    self.assertEqual(layer_1.dynamic_count, 4)

    sub_network.sub_layers["layer_1"] = None
    for _ in range(2):
      self.assertEqual(network.dynamic, False)
      self.assertEqual(layer_0.dynamic_count, 7)
      self.assertEqual(layer_1.dynamic_count, 4)

    layer_3 = AttrTrackingLayer()
    layer_3.stateful = True

    sub_network.sub_layers = None
    self.assertEqual(network.dynamic, False)
    self.assertEqual(network.stateful, False)

    # Test duplicate layers.
    sub_network.sub_layers = [layer_1, layer_1, layer_1, layer_3]
    self.assertEqual(network.dynamic, True)
    self.assertEqual(network.stateful, True)

    for _ in range(3):
      sub_network.sub_layers.pop()
      self.assertEqual(network.dynamic, True)
      self.assertEqual(network.stateful, False)

    sub_network.sub_layers.pop()
    self.assertEqual(network.dynamic, False)
    self.assertEqual(network.stateful, False)

  def test_compute_output_shape_cache(self):
    # See https://github.com/tensorflow/tensorflow/issues/32029.
    x = input_layer_lib.Input(shape=(None, 32))
    dense = layers.Dense(2)
    y = dense(x)
    network = functional.Functional(x, y, name='dense_network')

    for i in range(999, 1024):
      self.assertEqual(network.compute_output_shape((1, i, 32)), (1, i, 2))

  def test_2d_inputs_squeezed_to_1d(self):
    input_1d = input_layer_lib.Input(shape=())
    outputs = input_1d * 2.
    net = functional.Functional(input_1d, outputs)

    x = np.ones((10, 1))
    y = net(x)
    self.assertEqual(y.shape.rank, 1)

  def test_1d_inputs_expanded_to_2d(self):
    input_1d = input_layer_lib.Input(shape=(1,))
    outputs = input_1d * 2.
    net = functional.Functional(input_1d, outputs)

    x = np.ones((10,))
    y = net(x)
    self.assertEqual(y.shape.rank, 2)

  def test_training_passed_during_construction(self):

    class MyLayer(base_layer.Layer):

      def call(self, x, training=None):
        if training is None:
          return x * -1.0
        elif training:
          return x
        else:
          return x * 0.0

    my_layer = MyLayer()
    x = np.ones((1, 10))

    inputs = input_layer_lib.Input(10)
    outputs = my_layer(inputs, training=True)
    network = functional.Functional(inputs, outputs)

    # Hard-coded value passed during construction is respected.
    self.assertAllEqual(network(x, training=False), x)

    inputs = input_layer_lib.Input(10)
    outputs = my_layer(inputs, training=False)
    network = functional.Functional(inputs, outputs)

    network(x, training=True)
    # Hard-coded value passed during construction is respected.
    self.assertAllEqual(network(x, training=True), x * 0.0)

    inputs = input_layer_lib.Input(10)
    outputs = my_layer(inputs, training=None)
    network = functional.Functional(inputs, outputs)

    # `None` value passed during construction is overridden.
    self.assertAllEqual(network(x, training=True), x)
    # `None` value passed during construction is overridden.
    self.assertAllEqual(network(x, training=False), x * 0.0)

if __name__ == '__main__':
  test.main()
