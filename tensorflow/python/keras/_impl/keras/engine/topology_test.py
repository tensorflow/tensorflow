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

import os
import shutil

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.keras._impl import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

try:
  import yaml  # pylint:disable=g-import-not-at-top
except ImportError:
  yaml = None

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TopologyConstructionTest(test.TestCase):

  def test_get_updates_for(self):
    a = keras.layers.Input(shape=(2,))
    dense_layer = keras.layers.Dense(1)
    dense_layer.add_update(0, inputs=a)
    dense_layer.add_update(1, inputs=None)

    self.assertListEqual(dense_layer.get_updates_for(a), [0])
    self.assertListEqual(dense_layer.get_updates_for(None), [1])

  def test_get_losses_for(self):
    a = keras.layers.Input(shape=(2,))
    dense_layer = keras.layers.Dense(1)
    dense_layer.add_loss(0, inputs=a)
    dense_layer.add_loss(1, inputs=None)

    self.assertListEqual(dense_layer.get_losses_for(a), [0])
    self.assertListEqual(dense_layer.get_losses_for(None), [1])

  def test_trainable_weights(self):
    a = keras.layers.Input(shape=(2,))
    b = keras.layers.Dense(1)(a)
    model = keras.models.Model(a, b)

    weights = model.weights
    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

    model.trainable = True
    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.layers[1].trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

    # sequential model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=2))
    weights = model.weights

    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

    model.trainable = True
    self.assertListEqual(model.trainable_weights, weights)
    self.assertListEqual(model.non_trainable_weights, [])

    model.layers[0].trainable = False
    self.assertListEqual(model.trainable_weights, [])
    self.assertListEqual(model.non_trainable_weights, weights)

  def test_weight_loading(self):
    with self.test_session():
      a = keras.layers.Input(shape=(2,))
      x = keras.layers.Dense(3)(a)
      b = keras.layers.Dense(1)(x)
      model = keras.models.Model(a, b)

      x = np.random.random((3, 2))
      ref_y = model.predict(x)
      weights = model.get_weights()
      model.set_weights(weights)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      with self.assertRaises(ValueError):
        model.set_weights(weights[1:])
      with self.assertRaises(ValueError):
        model.set_weights(weights[::-1])

      if h5py is None:
        return  # Skip rest of test if H5py isn't available.

      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      h5_path = os.path.join(temp_dir, 'test.h5')
      model.save_weights(h5_path)
      model.load_weights(h5_path)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

      model.load_weights(h5_path, by_name=True)
      y = model.predict(x)
      self.assertAllClose(ref_y, y)

  def test_learning_phase(self):
    with self.test_session():
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      a_2 = keras.layers.Dense(16, name='dense_1')(a)
      dp = keras.layers.Dropout(0.5, name='dropout')
      b_2 = dp(b)

      self.assertFalse(a_2._uses_learning_phase)
      self.assertTrue(b_2._uses_learning_phase)

      # test merge
      m = keras.layers.concatenate([a_2, b_2])
      self.assertTrue(m._uses_learning_phase)

      # Test recursion
      model = keras.models.Model([a, b], [a_2, b_2])
      self.assertTrue(model.uses_learning_phase)

      c = keras.layers.Input(shape=(32,), name='input_c')
      d = keras.layers.Input(shape=(32,), name='input_d')

      c_2, b_2 = model([c, d])
      self.assertTrue(c_2._uses_learning_phase)
      self.assertTrue(b_2._uses_learning_phase)

      # try actually running graph
      fn = keras.backend.function(
          model.inputs + [keras.backend.learning_phase()], model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs_no_dp = fn([input_a_np, input_b_np, 0])
      fn_outputs_dp = fn([input_a_np, input_b_np, 1])
      # output a: nothing changes
      self.assertEqual(fn_outputs_no_dp[0].sum(), fn_outputs_dp[0].sum())
      # output b: dropout applied
      self.assertNotEqual(fn_outputs_no_dp[1].sum(), fn_outputs_dp[1].sum())

  def test_layer_call_arguments(self):
    # Test the ability to pass and serialize arguments to `call`.
    inp = keras.layers.Input(shape=(2,))
    x = keras.layers.Dense(3)(inp)
    x = keras.layers.Dropout(0.5)(x, training=True)
    model = keras.models.Model(inp, x)
    self.assertFalse(model.uses_learning_phase)

    # Test that argument is kept when applying the model
    inp2 = keras.layers.Input(shape=(2,))
    out2 = model(inp2)
    self.assertFalse(out2._uses_learning_phase)

    # Test that argument is kept after loading a model
    config = model.get_config()
    model = keras.models.Model.from_config(config)
    self.assertFalse(model.uses_learning_phase)

  def test_node_construction(self):
    # test basics
    a = keras.layers.Input(shape=(32,), name='input_a')
    b = keras.layers.Input(shape=(32,), name='input_b')

    with self.assertRaises(ValueError):
      _ = keras.layers.Input(shape=(32,), batch_shape=(10, 32))
    with self.assertRaises(ValueError):
      _ = keras.layers.Input(shape=(32,), unknown_kwarg=None)

    self.assertListEqual(a.get_shape().as_list(), [None, 32])
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

    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    self.assertEqual(len(dense._inbound_nodes), 2)
    self.assertEqual(len(dense._outbound_nodes), 0)
    self.assertListEqual(dense._inbound_nodes[0].inbound_layers, [a_layer])
    self.assertEqual(dense._inbound_nodes[0].outbound_layer, dense)
    self.assertListEqual(dense._inbound_nodes[1].inbound_layers, [b_layer])
    self.assertEqual(dense._inbound_nodes[1].outbound_layer, dense)
    self.assertListEqual(dense._inbound_nodes[0].input_tensors, [a])
    self.assertListEqual(dense._inbound_nodes[1].input_tensors, [b])

    # test layer properties
    test_layer = keras.layers.Dense(16, name='test_layer')
    a_test = test_layer(a)
    self.assertListEqual(test_layer.kernel.get_shape().as_list(), [32, 16])
    self.assertEqual(test_layer.input, a)
    self.assertEqual(test_layer.output, a_test)
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, (None, 16))

    self.assertEqual(dense.get_input_at(0), a)
    self.assertEqual(dense.get_input_at(1), b)
    self.assertEqual(dense.get_output_at(0), a_2)
    self.assertEqual(dense.get_output_at(1), b_2)
    self.assertEqual(dense.get_input_shape_at(0), (None, 32))
    self.assertEqual(dense.get_input_shape_at(1), (None, 32))
    self.assertEqual(dense.get_output_shape_at(0), (None, 16))
    self.assertEqual(dense.get_output_shape_at(1), (None, 16))
    self.assertEqual(dense.get_input_mask_at(0), None)
    self.assertEqual(dense.get_input_mask_at(1), None)
    self.assertEqual(dense.get_output_mask_at(0), None)
    self.assertEqual(dense.get_output_mask_at(1), None)

  def test_multi_input_layer(self):
    with self.test_session():
      # test multi-input layer
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      dense = keras.layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)

      merged = keras.layers.concatenate([a_2, b_2], name='merge')
      self.assertListEqual(merged.get_shape().as_list(), [None, 16 * 2])
      merge_layer, merge_node_index, merge_tensor_index = merged._keras_history

      self.assertEqual(merge_node_index, 0)
      self.assertEqual(merge_tensor_index, 0)

      self.assertEqual(len(merge_layer._inbound_nodes), 1)
      self.assertEqual(len(merge_layer._outbound_nodes), 0)

      self.assertEqual(len(merge_layer._inbound_nodes[0].input_tensors), 2)
      self.assertEqual(len(merge_layer._inbound_nodes[0].inbound_layers), 2)

      c = keras.layers.Dense(64, name='dense_2')(merged)
      d = keras.layers.Dense(5, name='dense_3')(c)

      model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')
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
      fn = keras.backend.function(model.inputs, model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 64), (10, 5)])

      # test get_source_inputs
      self.assertListEqual(keras.engine.topology.get_source_inputs(c), [a, b])

      # serialization / deserialization
      json_config = model.to_json()
      recreated_model = keras.models.model_from_json(json_config)
      recreated_model.compile('rmsprop', 'mse')

      self.assertListEqual([l.name for l in recreated_model.layers][2:],
                           ['dense_1', 'merge', 'dense_2', 'dense_3'])
      self.assertListEqual([l.name for l in recreated_model._input_layers],
                           ['input_a', 'input_b'])
      self.assertListEqual([l.name for l in recreated_model._output_layers],
                           ['dense_2', 'dense_3'])

      fn = keras.backend.function(recreated_model.inputs,
                                  recreated_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 64), (10, 5)])

  def test_recursion(self):
    with self.test_session():
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      dense = keras.layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = keras.layers.concatenate([a_2, b_2], name='merge')
      c = keras.layers.Dense(64, name='dense_2')(merged)
      d = keras.layers.Dense(5, name='dense_3')(c)

      model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

      e = keras.layers.Input(shape=(32,), name='input_e')
      f = keras.layers.Input(shape=(32,), name='input_f')
      g, h = model([e, f])

      self.assertListEqual(g.get_shape().as_list(), c.get_shape().as_list())
      self.assertListEqual(h.get_shape().as_list(), d.get_shape().as_list())

      # test separate manipulation of different layer outputs
      i = keras.layers.Dense(7, name='dense_4')(h)

      final_model = keras.models.Model(
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
      fn = keras.backend.function(final_model.inputs, final_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 7), (10, 64)])

      # test serialization
      model_config = final_model.get_config()
      recreated_model = keras.models.Model.from_config(model_config)

      fn = keras.backend.function(recreated_model.inputs,
                                  recreated_model.outputs)
      input_a_np = np.random.random((10, 32))
      input_b_np = np.random.random((10, 32))
      fn_outputs = fn([input_a_np, input_b_np])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 7), (10, 64)])

  def test_multi_input_multi_output_recursion(self):
    with self.test_session():
      # test multi-input multi-output
      a = keras.layers.Input(shape=(32,), name='input_a')
      b = keras.layers.Input(shape=(32,), name='input_b')

      dense = keras.layers.Dense(16, name='dense_1')
      a_2 = dense(a)
      b_2 = dense(b)
      merged = keras.layers.concatenate([a_2, b_2], name='merge')
      c = keras.layers.Dense(64, name='dense_2')(merged)
      d = keras.layers.Dense(5, name='dense_3')(c)

      model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

      j = keras.layers.Input(shape=(32,), name='input_j')
      k = keras.layers.Input(shape=(32,), name='input_k')
      _, n = model([j, k])

      o = keras.layers.Input(shape=(32,), name='input_o')
      p = keras.layers.Input(shape=(32,), name='input_p')
      q, _ = model([o, p])

      self.assertListEqual(n.get_shape().as_list(), [None, 5])
      self.assertListEqual(q.get_shape().as_list(), [None, 64])
      s = keras.layers.concatenate([n, q], name='merge_nq')
      self.assertListEqual(s.get_shape().as_list(), [None, 64 + 5])

      # test with single output as 1-elem list
      multi_io_model = keras.models.Model([j, k, o, p], [s])

      fn = keras.backend.function(multi_io_model.inputs, multi_io_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      # test with single output as tensor
      multi_io_model = keras.models.Model([j, k, o, p], s)

      fn = keras.backend.function(multi_io_model.inputs, multi_io_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      # note that the output of the function will still be a 1-elem list
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      # test serialization
      model_config = multi_io_model.get_config()
      recreated_model = keras.models.Model.from_config(model_config)

      fn = keras.backend.function(recreated_model.inputs,
                                  recreated_model.outputs)
      fn_outputs = fn([
          np.random.random((10, 32)), np.random.random((10, 32)),
          np.random.random((10, 32)), np.random.random((10, 32))
      ])
      # note that the output of the function will still be a 1-elem list
      self.assertListEqual([x.shape for x in fn_outputs], [(10, 69)])

      config = model.get_config()
      keras.models.Model.from_config(config)

      model.summary()
      json_str = model.to_json()
      keras.models.model_from_json(json_str)

      if yaml is not None:
        yaml_str = model.to_yaml()
        keras.models.model_from_yaml(yaml_str)

  def test_invalid_graphs(self):
    a = keras.layers.Input(shape=(32,), name='input_a')
    b = keras.layers.Input(shape=(32,), name='input_b')

    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)
    merged = keras.layers.concatenate([a_2, b_2], name='merge')
    c = keras.layers.Dense(64, name='dense_2')(merged)
    d = keras.layers.Dense(5, name='dense_3')(c)

    model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

    # input is not an Input tensor
    j = keras.layers.Input(shape=(32,), name='input_j')
    j = keras.layers.Dense(32)(j)
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    with self.assertRaises(Exception):
      keras.models.Model([j, k], [m, n])

    # disconnected graph
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      keras.models.Model([j], [m, n])

    # redundant outputs
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    keras.models.Model([j, k], [m, n, n])

    # redundant inputs
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      keras.models.Model([j, k, j], [m, n])

    # i have not idea what I'm doing: garbage as inputs/outputs
    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with self.assertRaises(Exception):
      keras.models.Model([j, k], [m, n, 0])

  def test_raw_tf_compatibility(self):
    # test calling layers/models on TF tensors
    a = keras.layers.Input(shape=(32,), name='input_a')
    b = keras.layers.Input(shape=(32,), name='input_b')

    dense = keras.layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)
    merged = keras.layers.concatenate([a_2, b_2], name='merge')
    c = keras.layers.Dense(64, name='dense_2')(merged)
    d = keras.layers.Dense(5, name='dense_3')(c)

    model = keras.models.Model(inputs=[a, b], outputs=[c, d], name='model')

    j = keras.layers.Input(shape=(32,), name='input_j')
    k = keras.layers.Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    tf_model = keras.models.Model([j, k], [m, n])

    j_tf = array_ops.placeholder(dtype=dtypes.float32, shape=(None, 32))
    k_tf = array_ops.placeholder(dtype=dtypes.float32, shape=(None, 32))
    m_tf, n_tf = tf_model([j_tf, k_tf])
    self.assertListEqual(m_tf.get_shape().as_list(), [None, 64])
    self.assertListEqual(n_tf.get_shape().as_list(), [None, 5])

    # test merge
    keras.layers.concatenate([j_tf, k_tf], axis=1)
    keras.layers.add([j_tf, k_tf])

    # test tensor input
    x = array_ops.placeholder(shape=(None, 2), dtype=dtypes.float32)
    keras.layers.InputLayer(input_tensor=x)

    x = keras.layers.Input(tensor=x)
    keras.layers.Dense(2)(x)

  def test_basic_masking(self):
    a = keras.layers.Input(shape=(10, 32), name='input_a')
    b = keras.layers.Masking()(a)
    model = keras.models.Model(a, b)
    self.assertEqual(model.output_mask.get_shape().as_list(), [None, 10])

  def test_weight_preprocessing(self):
    input_dim = 3
    output_dim = 3
    size = 2
    cases = [
        [
            (keras.layers.Bidirectional(keras.layers.SimpleRNN(2))),
            [np.random.random((2, 1)), np.random.random((2, 1))],
            (None, 3, 2),
        ],
        [
            (keras.layers.TimeDistributed(keras.layers.Dense(1))),
            [np.random.random((2, 1)), np.random.random((1,))],
            (None, 3, 2),
        ],
        [
            (keras.layers.Conv1D(output_dim, size, use_bias=False)),
            [np.random.random((output_dim, input_dim, size, 1))],
            (None, 4, input_dim),
        ],
        [
            (keras.layers.Conv2D(output_dim, size,
                                 use_bias=False, data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size))],
            (None, input_dim, 4, 4),
        ],
        [
            (keras.layers.Conv2DTranspose(output_dim, size,
                                          use_bias=False,
                                          data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size))],
            (None, input_dim, 4, 4),
        ],
        [
            (keras.layers.Conv2DTranspose(output_dim, size,
                                          use_bias=False,
                                          data_format='channels_last')),
            [np.random.random((size, size, input_dim, output_dim))],
            (None, 4, 4, input_dim),
        ],
        [
            (keras.layers.Conv3D(output_dim, size,
                                 use_bias=False, data_format='channels_first')),
            [np.random.random((output_dim, input_dim, size, size, size))],
            (None, input_dim, 4, 4, 4),
        ],
        [
            (keras.layers.GRU(output_dim)),
            [np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,))],
            (None, 4, input_dim),
        ],
        [
            (keras.layers.LSTM(output_dim)),
            [np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,)),
             np.random.random((input_dim, output_dim)),
             np.random.random((output_dim, output_dim)),
             np.random.random((output_dim,))],
            (None, 4, input_dim),
        ],
    ]
    for layer, weights, input_shape in cases:
      layer.build(input_shape)
      _ = keras.engine.topology.preprocess_weights_for_loading(
          layer, weights, original_keras_version='1')

    model = keras.models.Sequential([keras.layers.Dense(2, input_dim=2)])
    _ = keras.engine.topology.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

    x = keras.Input((2,))
    y = keras.layers.Dense(2)(x)
    model = keras.models.Model(x, y)
    _ = keras.engine.topology.preprocess_weights_for_loading(
        model, model.weights, original_keras_version='1')

  def test_layer_sharing_at_heterogenous_depth(self):
    with self.test_session():
      x_val = np.random.random((10, 5))

      x = keras.Input(shape=(5,))
      a = keras.layers.Dense(5, name='A')
      b = keras.layers.Dense(5, name='B')
      output = a(b(a(b(x))))
      m = keras.models.Model(x, output)

      output_val = m.predict(x_val)

      config = m.get_config()
      weights = m.get_weights()

      m2 = keras.models.Model.from_config(config)
      m2.set_weights(weights)

      output_val_2 = m2.predict(x_val)
      self.assertAllClose(output_val, output_val_2, atol=1e-6)

  def test_layer_sharing_at_heterogenous_depth_with_concat(self):
    with self.test_session():
      input_shape = (16, 9, 3)
      input_layer = keras.Input(shape=input_shape)

      a = keras.layers.Dense(3, name='dense_A')
      b = keras.layers.Dense(3, name='dense_B')
      c = keras.layers.Dense(3, name='dense_C')

      x1 = b(a(input_layer))
      x2 = a(c(input_layer))
      output = keras.layers.concatenate([x1, x2])

      m = keras.models.Model(inputs=input_layer, outputs=output)

      x_val = np.random.random((10, 16, 9, 3))
      output_val = m.predict(x_val)

      config = m.get_config()
      weights = m.get_weights()

      m2 = keras.models.Model.from_config(config)
      m2.set_weights(weights)

      output_val_2 = m2.predict(x_val)
      self.assertAllClose(output_val, output_val_2, atol=1e-6)

if __name__ == '__main__':
  test.main()
