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
"""Tests for tensorflow.python.client.graph_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import test
from tensorflow.python.training.saver import export_meta_graph


class ConvertVariablesToConstantsTest(test.TestCase):

  def _get_tensors(self, sess, tensor_list):
    """Returns a list of Tensor objects from the Session."""
    return [
        sess.graph.get_tensor_by_name(tensor.name) for tensor in tensor_list
    ]

  def _get_tensor_names(self, tensors):
    """Returns a list of string names for the tensors specified."""
    return [tensor.name.split(":")[0] for tensor in tensors]

  def _evaluate_graph_def(self, graph_def, inputs, outputs, input_data):
    """Evaluates the GraphDef using Sessions."""
    with ops.Graph().as_default() as graph:
      importer.import_graph_def(graph_def, name="")
      sess = session.Session(graph=graph)

    input_tensors = self._get_tensors(sess, inputs)
    output_tensors = self._get_tensors(sess, outputs)
    return sess.run(
        output_tensors, feed_dict=dict(zip(input_tensors, input_data)))

  def _ensure_no_variables_in_graph(self, graph_def):
    """Ensures there are no variables in the graph."""
    for node in graph_def.node:
      self.assertNotIn(
          node.op, ["Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"])

  def _test_converted_keras_model(self, model, constant_graph_def, input_data):
    """Compares the converted Keras model."""
    expected_value = model.predict(input_data)
    actual_value = self._evaluate_graph_def(constant_graph_def, model.inputs,
                                            model.outputs, [input_data])
    np.testing.assert_almost_equal(np.array([expected_value]), actual_value, 5)

  def _inline_functions(self, graph_def, arrays):
    meta_graph = export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for name in arrays:
      fetch_collection.node_list.value.append(name)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

    # Initialize RewriterConfig with everything disabled except function
    # inlining.
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.optimizers.append("function")
    return tf_optimizer.OptimizeGraph(config, meta_graph)

  def testWithEmbeddings(self):
    """Freezes a graph with embeddings."""
    state_input = keras.layers.Input(
        shape=(1,), name="state_input", dtype="int32")
    output = keras.layers.Embedding(
        output_dim=16, input_dim=100, input_length=1, name="state")(
            state_input)
    model = keras.models.Model(inputs=[state_input], outputs=[output])
    model.compile(
        loss={"state": "sparse_categorical_crossentropy"}, optimizer="adam")

    # Freeze the graph.
    sess = keras.backend.get_session()
    variable_graph_def = sess.graph_def
    output_tensor = self._get_tensor_names(model.outputs)
    constant_graph_def = graph_util.convert_variables_to_constants(
        sess, variable_graph_def, output_tensor)

    # Validate converted graph.
    input_data = np.array(np.random.random_sample([1, 1]), dtype=np.int32)
    self._ensure_no_variables_in_graph(constant_graph_def)
    self._test_converted_keras_model(model, constant_graph_def, input_data)

  def testKerasBatchNorm(self):
    """Freezes a graph with Keras batch norm."""
    inputs = keras.layers.Input(shape=(128, 128, 1))
    batch_norm = keras.layers.BatchNormalization()(inputs)
    model = keras.models.Model(inputs, batch_norm, name="test")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    tensor_names = [tensor.name for tensor in model.inputs + model.outputs]

    # Freeze the graph.
    sess = keras.backend.get_session()
    variable_graph_def = sess.graph_def
    variable_graph_def = self._inline_functions(variable_graph_def,
                                                tensor_names)
    output_tensor = self._get_tensor_names(model.outputs)
    constant_graph_def = graph_util.convert_variables_to_constants(
        sess, variable_graph_def, output_tensor)

    # Validate converted graph.
    input_data = np.array(
        np.random.random_sample([1, 128, 128, 1]), dtype=np.int32)
    self._ensure_no_variables_in_graph(constant_graph_def)
    self._test_converted_keras_model(model, constant_graph_def, input_data)

  def testLSTM(self):
    """Freezes a Keras LSTM."""
    model = keras.models.Sequential(
        [keras.layers.LSTM(units=10, input_shape=(10, 10))])
    tensor_names = [tensor.name for tensor in model.inputs + model.outputs]

    # Freeze the model.
    sess = keras.backend.get_session()
    variable_graph_def = sess.graph_def
    variable_graph_def = self._inline_functions(variable_graph_def,
                                                tensor_names)
    output_tensor = self._get_tensor_names(model.outputs)
    constant_graph_def = graph_util.convert_variables_to_constants(
        sess, variable_graph_def, output_tensor)

    # Validate converted graph.
    input_data = np.array(np.random.random_sample([10, 10, 10]), dtype=np.int32)
    self._ensure_no_variables_in_graph(constant_graph_def)
    self._test_converted_keras_model(model, constant_graph_def, input_data)


if __name__ == "__main__":
  ops.disable_eager_execution()
  test.main()
