# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lookup ops."""
import os
import tempfile
import unittest

from absl.testing import parameterized
import numpy as np
import six

from tensorflow.python import tf2
from tensorflow.python.checkpoint import checkpoint as trackable
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


class BaseLookupTableTest(test.TestCase):

  def getHashTable(self):
    if tf2.enabled():
      return lookup_ops.StaticHashTable
    else:
      return lookup_ops.StaticHashTableV1

  def getVocabularyTable(self):
    if tf2.enabled():
      return lookup_ops.StaticVocabularyTable
    else:
      return lookup_ops.StaticVocabularyTableV1

  def initialize_table(self, table):
    if not tf2.enabled():
      self.evaluate(table.initializer)


SKIP_ANONYMOUS_IN_TF1_REASON = (
    "In v1 graph mode, each self.evaluate call will execute the handle "
    "creation op (e.g. AnonymousHashTable) which will create a new table "
    "resource unrelated to other self.evaluate calls, so we can't test "
    "anonymous resources with self.evaluate ."
)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class StaticHashTableTest(BaseLookupTableTest, parameterized.TestCase):

  def testStaticHashTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertEqual(table._is_anonymous, is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

    exported_keys_tensor, exported_values_tensor = table.export()

    self.assertItemsEqual([b"brain", b"salad", b"surgery"],
                          self.evaluate(exported_keys_tensor))
    self.assertItemsEqual([0, 1, 2], self.evaluate(exported_values_tensor))

  def testStaticHashTableFindHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testStaticHashTableInitWithPythonArrays(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = ["brain", "salad", "surgery"]
    values = [0, 1, 2]
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            keys, values, value_dtype=dtypes.int64),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableInitWithNumPyArrays(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = np.array(["brain", "salad", "surgery"], dtype=np.str_)
    values = np.array([0, 1, 2], dtype=np.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testMultipleStaticHashTables(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)

    table1 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    table2 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    table3 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table1)
    self.initialize_table(table2)
    self.initialize_table(table3)
    self.assertAllEqual(3, self.evaluate(table1.size()))
    self.assertAllEqual(3, self.evaluate(table2.size()))
    self.assertAllEqual(3, self.evaluate(table3.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output1 = table1.lookup(input_string)
    output2 = table2.lookup(input_string)
    output3 = table3.lookup(input_string)

    out1, out2, out3 = self.evaluate([output1, output2, output3])
    self.assertAllEqual([0, 1, -1], out1)
    self.assertAllEqual([0, 1, -1], out2)
    self.assertAllEqual([0, 1, -1], out3)

  def testStaticHashTableWithTensorDefault(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableGetItem(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table[input_string]

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableWithSparseTensorInput(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_indices = [[0, 0], [0, 1], [1, 0]]
    sp_shape = [2, 2]
    input_tensor = sparse_tensor.SparseTensor(
        constant_op.constant(sp_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "tank"]),
        constant_op.constant(sp_shape, dtypes.int64))
    output = table.lookup(input_tensor)

    out_indices, out_values, out_shape = self.evaluate(output)

    self.assertAllEqual([0, 1, -1], out_values)
    self.assertAllEqual(sp_indices, out_indices)
    self.assertAllEqual(sp_shape, out_shape)

  def testStaticHashTableWithRaggedTensorInput(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    row_splits = [0, 2, 3]
    input_tensor = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "tank"]),
        constant_op.constant(row_splits, dtypes.int64))
    output = table.lookup(input_tensor)

    out = self.evaluate(output)

    self.assertAllEqual([0, 1, -1], out.values)
    self.assertAllEqual(row_splits, out.row_splits)

  def testSignatureMismatch(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    # Ref types do not produce a lookup signature mismatch.
    input_string_ref = variables.Variable("brain")
    self.evaluate(input_string_ref.initializer)
    self.assertEqual(0, self.evaluate(table.lookup(input_string_ref)))

    input_string = constant_op.constant([1, 2, 3], dtypes.int64)
    with self.assertRaises(TypeError):
      table.lookup(input_string)

    with self.assertRaises(TypeError):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          "UNK",
          experimental_is_anonymous=is_anonymous)

  def testDTypes(self, is_anonymous):
    default_val = -1
    with self.assertRaises(TypeError):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(["a"], [1], [dtypes.string],
                                               dtypes.int64),
          default_val,
          experimental_is_anonymous=is_anonymous)

  @test_util.run_v1_only("(Cached) Sessions not available in TF2.0")
  def testNotInitialized(self, is_anonymous):
    with self.cached_session():
      default_val = -1
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(["a"], [1],
                                               value_dtype=dtypes.int64),
          default_val,
          experimental_is_anonymous=is_anonymous)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        self.evaluate(output)

  @test_util.run_v1_only("(Cached) Sessions not available in TF2.0")
  def testInitializeTwice(self, is_anonymous):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)
      # Make sure that initializing twice doesn't throw any errors.
      self.initialize_table(table)

  def testInitializationWithInvalidDimensions(self, is_anonymous):
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)

    raised_error = ValueError
    if context.executing_eagerly():
      raised_error = errors_impl.InvalidArgumentError
    with self.assertRaises(raised_error):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          default_val,
          experimental_is_anonymous=is_anonymous)

  @test_util.run_v1_only("Sessions not available in TF2.0")
  def testMultipleSessions(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Start a server
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1",
        experimental_is_anonymous=is_anonymous)

    # Init the table in the first session.
    with session1:
      self.initialize_table(table)
      self.assertAllEqual(3, self.evaluate(table.size()))

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      self.evaluate(table.initializer)
      self.assertAllEqual(3, self.evaluate(table.size()))

  @test_util.run_v2_only
  def testImportedHashTable(self, is_anonymous):
    g = ops.Graph()
    with g.as_default():
      t = lookup_ops.StaticHashTable(
          lookup_ops.KeyValueTensorInitializer(["a"], [1]),
          2)
      init_op = t._init_op
      op = t.lookup(ops.convert_to_tensor(["a"]))
      meta_graph = saver.export_meta_graph()

    def f():
      saver.import_meta_graph(meta_graph)
      return ops.get_default_graph().get_tensor_by_name(op.name)

    wrapped = wrap_function.wrap_function(f, [])
    pruned_init_fn = wrapped.prune(
        (), [wrapped.graph.get_operation_by_name(init_op.name)])
    self.evaluate(pruned_init_fn())
    self.assertAllEqual([1], wrapped())

  def testStaticHashTableInt32String(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = "n/a"
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_tensor = constant_op.constant([0, 1, -1])
    output = table.lookup(input_tensor)

    result = self.evaluate(output)
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)

  def testTableUseInFunction(self, is_anonymous):
    if not context.executing_eagerly():
      self.skipTest("Only Eager mode test.")
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        "n/a",
        experimental_is_anonymous=is_anonymous)

    @function.defun()
    def lookup_table_func(k):
      return table.lookup(k)

    result = lookup_table_func(constant_op.constant([0, 1, -1]))
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)
    result = lookup_table_func(constant_op.constant([2, -1, 1]))
    self.assertAllEqual([b"surgery", b"n/a", b"salad"], result)

  def testTableCreatedInFunction(self, is_anonymous):
    if not context.executing_eagerly():
      self.skipTest("Only Eager mode test.")
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])

    @function.defun()
    def lookup_table_func(k):
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          "n/a",
          experimental_is_anonymous=is_anonymous)
      return table.lookup(k)

    result = lookup_table_func(constant_op.constant([0, 1, -1]))
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)
    result = lookup_table_func(constant_op.constant([2, -1, 1]))
    self.assertAllEqual([b"surgery", b"n/a", b"salad"], result)

  def testTwoTablesInControlFlow(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([1, 2, 3], dtypes.int32)
    values = constant_op.constant([5, 10, 15], dtypes.int32)

    def table_func1(x):
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          -1,
          experimental_is_anonymous=is_anonymous)
      return table.lookup(x)

    elems = np.array([2, 4, 1], dtype=np.int32)
    result1 = map_fn.map_fn(table_func1, elems, dtype=dtypes.int32)

    def table_func2(x):
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          -1,
          experimental_is_anonymous=is_anonymous)
      return table.lookup(x)

    elems = np.array([2, 4, 1], dtype=np.int32)
    result2 = map_fn.map_fn(table_func2, elems, dtype=dtypes.int32)

    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([10, -1, 5], self.evaluate(result1))
    self.assertAllEqual([10, -1, 5], self.evaluate(result2))

  @test_util.enable_control_flow_v2
  def testLookupTableInWhileV2(self, is_anonymous):
    lookup = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)

    beta = variables.Variable(1.0, trainable=True)

    @def_function.function
    def get_loss(unused_beta):
      return map_fn.map_fn(
          lookup.lookup,
          constant_op.constant([2, 3], dtype=dtypes.int64),
          dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      loss = get_loss(beta)

    self.assertIsNone(tape.gradient(loss, beta))

  @test_util.enable_control_flow_v2
  def testLookupTableInCondV2(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    lookup = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)

    beta = variables.Variable(1.0, trainable=True)

    @def_function.function
    def get_loss(beta):

      def true_fn():
        return lookup.lookup(constant_op.constant(2, dtype=dtypes.int64))

      def false_fn():
        return constant_op.constant(0, dtype=dtypes.float32)

      return beta * control_flow_ops.cond(
          constant_op.constant(True), true_fn=true_fn, false_fn=false_fn)

    with backprop.GradientTape() as tape:
      loss = get_loss(beta)
    grad = tape.gradient(loss, beta)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual(grad, -10.)

  def testExportShapeInference(self, is_anonymous):
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_value,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.lookup(12), 1)
    self.assertEqual(root.lookup(10), -1)
    self.assertLen(root.table.export()[0], 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class KeyValueTensorInitializerTest(BaseLookupTableTest):

  def test_string(self, is_anonymous):
    init = lookup_ops.KeyValueTensorInitializer(
        ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_multiple_tables(self, is_anonymous):
    with ops.name_scope("table_scope"):
      init1 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table1 = self.getHashTable()(
          init1, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table", table1.name)
        self.assertEqual("table_scope/hash_table",
                         table1.resource_handle.op.name)
      init2 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table2 = self.getHashTable()(
          init2, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table_1", table2.name)
        self.assertEqual("table_scope/hash_table_1",
                         table2.resource_handle.op.name)

  def test_int64(self, is_anonymous):
    init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int64, dtypes.int64)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_int32(self, is_anonymous):
    init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int32, dtypes.int64)
    with self.assertRaises(errors_impl.OpError):
      table = self.getHashTable()(
          init, default_value=-1, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class InitializeTableFromFileOpTest(BaseLookupTableTest):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testInitializeStringTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("one_column_1.txt")
    default_value = -1
    init = lookup_ops.TextFileInitializer(
        vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
        dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
    self.assertIn("one_column_1.txt_-2_-1", init._shared_name)
    table = self.getHashTable()(
        init, default_value, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    output = table.lookup(constant_op.constant(["brain", "salad", "tank"]))

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testInitializeInt64Table(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile(
        "one_column_int64.txt", values=("42", "1", "-1000"))

    with self.cached_session():
      default_value = -1
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_int64.txt_-2_-1", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      output = table.lookup(
          constant_op.constant((42, 1, 11), dtype=dtypes.int64))

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testInitializeIndexTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("one_column_2.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup_ops.TextFileIndex.LINE_NUMBER
      value_index = lookup_ops.TextFileIndex.WHOLE_LINE
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.int64, key_index, dtypes.string, value_index)
      self.assertIn("one_column_2.txt_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      output = table.lookup(input_values)

      result = self.evaluate(output)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], result)

  def testMultiColumn(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 1
      value_index = 2

      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertIn("three_columns.txt_1_2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([1, 5, 6], result)

  def testInvalidDataTypeInMultiColumn(self, is_anonymous):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 2
      value_index = 1
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertIn("three_columns.txt_2_1", init._shared_name)
      with self.assertRaisesOpError("is not a valid"):
        table = self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table)

  def testInvalidDataType(self, is_anonymous):
    vocabulary_file = self._createVocabFile("one_column_3.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup_ops.TextFileIndex.WHOLE_LINE
      value_index = lookup_ops.TextFileIndex.LINE_NUMBER

      with self.assertRaises(ValueError):
        init = lookup_ops.TextFileInitializer(vocabulary_file, dtypes.int64,
                                              key_index, dtypes.string,
                                              value_index)
        self.assertIn("one_column_3.txt_-2_-1", init._shared_name)
        self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)

  def testInvalidIndex(self, is_anonymous):
    vocabulary_file = self._createVocabFile("one_column_4.txt")
    with self.cached_session():
      default_value = -1
      key_index = 1  # second column of the line
      value_index = lookup_ops.TextFileIndex.LINE_NUMBER
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertIn("one_column_4.txt_1_-1", init._shared_name)

      with self.assertRaisesOpError("Invalid number of columns"):
        table = self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table)

  def testInitializeSameTableWithMultipleNodes(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("one_column_5.txt")

    with self.cached_session():
      default_value = -1
      init1 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_5.txt_-2_-1", init1._shared_name)
      table1 = self.getHashTable()(
          init1, default_value, experimental_is_anonymous=is_anonymous)
      init2 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_5.txt_-2_-1", init2._shared_name)
      table2 = self.getHashTable()(
          init2, default_value, experimental_is_anonymous=is_anonymous)
      init3 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_5.txt_-2_-1", init3._shared_name)
      table3 = self.getHashTable()(
          init3, default_value, experimental_is_anonymous=is_anonymous)

      self.evaluate(lookup_ops.tables_initializer())

      input_string = constant_op.constant(["brain", "salad", "tank"])

      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testInitializeTableWithNoFilename(self, is_anonymous):
    with self.cached_session():
      default_value = -1
      with self.assertRaises(ValueError):
        self.getHashTable()(
            lookup_ops.TextFileInitializer(
                "", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
            default_value,
            experimental_is_anonymous=is_anonymous)

  def testInitializeWithVocabSize(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      vocabulary_file1 = self._createVocabFile("one_column6.txt")
      init1 = lookup_ops.TextFileInitializer(
          vocabulary_file1,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertIn("one_column6.txt_3_-2_-1", init1._shared_name)
      table1 = self.getHashTable()(
          init1, default_value, experimental_is_anonymous=is_anonymous)

      # Initialize from file.
      self.initialize_table(table1)
      self.assertEqual(vocab_size, self.evaluate(table1.size()))

      vocabulary_file2 = self._createVocabFile("one_column7.txt")
      vocab_size = 5
      init2 = lookup_ops.TextFileInitializer(
          vocabulary_file2,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertIn("one_column7.txt_5_-2_-1", init2._shared_name)
      with self.assertRaisesOpError("Invalid vocab_size"):
        table2 = self.getHashTable()(
            init2, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table2)

      vocab_size = 1
      vocabulary_file3 = self._createVocabFile("one_column3.txt")
      init3 = lookup_ops.TextFileInitializer(
          vocabulary_file3,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertIn("one_column3.txt_1_-2_-1", init3._shared_name)
      table3 = self.getHashTable()(
          init3, default_value, experimental_is_anonymous=is_anonymous)

      # Smaller vocab size reads only vocab_size records.
      self.initialize_table(table3)
      self.assertEqual(vocab_size, self.evaluate(table3.size()))

  @test_util.run_v1_only("placeholder usage")
  def testFeedVocabularyName(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

    with self.cached_session():
      default_value = -1
      init = lookup_ops.TextFileInitializer(
          "old_file.txt", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("old_file.txt_-2_-1", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)

      # Initialize with non existing file (old_file.txt) should fail.
      # TODO(yleon): Update message, which might change per FileSystem.
      with self.assertRaisesOpError("old_file.txt"):
        self.evaluate(table.initializer)

      # Initialize the model feeding the vocabulary file.
      filenames = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
      table.initializer.run(feed_dict={filenames[0]: vocabulary_file})

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testInvalidFilenames(self, is_anonymous):
    vocabulary_file = self._createVocabFile("filename_shape.txt")

    with self.cached_session():
      default_value = -1

      # Invalid data type
      other_type = constant_op.constant(1)
      with self.assertRaises(Exception) as cm:
        self.getHashTable()(
            lookup_ops.TextFileInitializer(
                other_type, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
            default_value,
            experimental_is_anonymous=is_anonymous)
      self.assertIsInstance(cm.exception, (ValueError, TypeError))

      # Non-scalar filename
      filenames = constant_op.constant([vocabulary_file, vocabulary_file])
      if not context.executing_eagerly():
        with self.assertRaises(Exception) as cm:
          self.getHashTable()(
              lookup_ops.TextFileInitializer(
                  filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                  dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
              default_value,
              experimental_is_anonymous=is_anonymous)
        self.assertIsInstance(cm.exception, (ValueError, TypeError))
      else:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          self.getHashTable()(
              lookup_ops.TextFileInitializer(
                  filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                  dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
              default_value,
              experimental_is_anonymous=is_anonymous)

  def testIdToStringTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = "UNK"
      vocab_size = 3
      init = lookup_ops.TextFileStringTableInitializer(
          vocab_file, vocab_size=vocab_size)
      self.assertTrue("feat_to_id_1.txt_3_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)

      self.initialize_table(table)

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)

      out = table.lookup(input_values)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"],
                          self.evaluate(out))
      self.assertEqual(vocab_size, self.evaluate(table.size()))

  def testStringToIdTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_2.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      init = lookup_ops.TextFileIdTableInitializer(
          vocab_file, vocab_size=vocab_size)
      self.assertTrue("feat_to_id_2.txt_3_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, -1], self.evaluate(out))
      self.assertEqual(vocab_size, self.evaluate(table.size()))

  def testInt64ToIdTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile(
        "feat_to_id_3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      init = lookup_ops.TextFileIdTableInitializer(
          vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64)
      self.assertTrue("feat_to_id_3.txt_3_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      out = table.lookup(
          constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64))
      self.assertAllEqual((0, 1, 2, -1), self.evaluate(out))
      self.assertEqual(vocab_size, self.evaluate(table.size()))


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class StaticVocabularyTableTest(BaseLookupTableTest):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testStringStaticVocabularyTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out = table.lookup(input_string)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testStaticVocabularyTableGetItem(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out = table[input_string]
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt32StaticVocabularyTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
        oov_buckets,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt64StaticVocabularyTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testStringStaticVocabularyTableNoInitializer(self, is_anonymous):
    oov_buckets = 5

    # Set a table that only uses hash buckets, for each input value returns
    # an id calculated by fingerprint("input") mod oov_buckets.
    table = self.getVocabularyTable()(
        None, oov_buckets, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    values = constant_op.constant(("brain", "salad", "surgery"))

    out = table.lookup(values)
    self.assertAllEqual(
        [
            3,  # fingerprint("brain") mod 5.
            1,  # fingerprint("salad") mod 5.
            4  # fingerprint("surgery") mod 5
        ],
        self.evaluate(out))
    self.assertEqual(oov_buckets, self.evaluate(table.size()))

  def testStaticVocabularyTableWithMultipleInitializers(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    vocab_size = 3
    oov_buckets = 3

    init = lookup_ops.TextFileIdTableInitializer(
        vocab_file, vocab_size=vocab_size)
    table1 = self.getVocabularyTable()(
        init,
        oov_buckets,
        name="table1",
        experimental_is_anonymous=is_anonymous)

    table2 = self.getVocabularyTable()(
        init,
        oov_buckets,
        name="table2",
        experimental_is_anonymous=is_anonymous)

    self.evaluate(lookup_ops.tables_initializer())

    input_string = constant_op.constant(
        ["fruit", "brain", "salad", "surgery", "UNK"])

    out1 = table1.lookup(input_string)
    out2 = table2.lookup(input_string)

    out1, out2 = self.evaluate([out1, out2])
    self.assertAllEqual([5, 0, 1, 2, 5], out1)
    self.assertAllEqual([5, 0, 1, 2, 5], out2)
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testStaticVocabularyTableInitializationAcrossSessions(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    with self.cached_session():
      vocab_size = 3
      oov_buckets = 1
      table1 = self.getVocabularyTable()(
          lookup_ops.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          oov_buckets,
          experimental_is_anonymous=is_anonymous)

      self.initialize_table(table1)

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string_1)

      self.assertAllEqual([0, 1, 2, 3], self.evaluate(out1))
      self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))

    with self.cached_session():
      vocab_size = 3
      oov_buckets = 1

      # Underlying lookup table already initialized in previous session.
      # No need to initialize table2
      table2 = self.getVocabularyTable()(
          lookup_ops.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          oov_buckets,
          experimental_is_anonymous=is_anonymous)

      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out2 = table2.lookup(input_string_2)

      self.assertAllEqual([3, 1, 3], self.evaluate(out2))
      self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testStaticVocabularyTableAssetTracking(self, is_anonymous):
    vocab_file = self._createVocabFile("vocab.txt")
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)
    objects = checkpoint_util.list_objects(graph_view.ObjectGraphView(table))
    assets = list(filter(lambda obj: isinstance(obj, asset.Asset), objects))
    self.assertLen(assets, 1)
    self.assertEqual(
        self.evaluate(assets[0].asset_path), compat.as_bytes(vocab_file))

  def testSparseTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testRaggedTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt32SparseTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32RaggedTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt64SparseTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64RaggedTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testStaticVocabularyTableNoInnerTable(self, is_anonymous):
    table = self.getVocabularyTable()(
        None, num_oov_buckets=1, experimental_is_anonymous=is_anonymous)
    self.assertIsNone(table.resource_handle)

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("11", "12", "13"))
    vocab_size = 3
    oov_buckets = 1
    root.table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 4)
    self.assertEqual(root.lookup(12), 1)
    self.assertEqual(root.lookup(10), 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 4)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), 3)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class DenseHashTableOpTest(test.TestCase):

  def testBasic(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant([12, 15], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([11, 12, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, -1, -1], result)

  def testGetItem(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))

    input_string = constant_op.constant([11, 12, 15], dtypes.int64)
    output = table[input_string]
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testBasicBool(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([True, True, True, True], dtypes.bool)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.bool,
        default_value=False,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant([11, 15], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([11, 12, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([False, True, False], result)

  def testSameEmptyAndDeletedKey(self, is_anonymous):
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Empty and deleted keys"):
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=42,
          deleted_key=42,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table.size()))

  @test_util.run_v1_only("uses placeholders")
  def testLookupUnknownShape(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1,
          experimental_is_anonymous=is_anonymous)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      placeholder_keys = array_ops.placeholder(dtypes.int64)
      output = table.lookup(placeholder_keys)
      self.assertAllEqual(None, output.get_shape())
      result = output.eval({placeholder_keys: [11, 12, 15]})
      self.assertAllEqual([0, 1, -1], result)

  def testMapStringToFloat(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant(["a", "b", "c", "d"], dtypes.string)
    values = constant_op.constant([0.0, 1.1, 2.2, 3.3], dtypes.float32)
    default_value = constant_op.constant(-1.5, dtypes.float32)
    table = lookup_ops.DenseHashTable(
        dtypes.string,
        dtypes.float32,
        default_value=default_value,
        empty_key="",
        deleted_key="$",
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["b", "e"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "d", "e"], dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([4], output.get_shape())

    result = self.evaluate(output)
    self.assertAllClose([0, -1.5, 3.3, -1.5], result)

  def testMapInt64ToFloat(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    for float_dtype in [dtypes.float32, dtypes.float64]:
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0.0, 1.1, 2.2, 3.3], float_dtype)
      default_value = constant_op.constant(-1.5, float_dtype)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          float_dtype,
          default_value=default_value,
          empty_key=0,
          deleted_key=-1,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant([12, 15], dtypes.int64)
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([11, 12, 14, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([4], output.get_shape())

      result = self.evaluate(output)
      self.assertAllClose([0, -1.5, 3.3, -1.5], result)

  def testVectorValues(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]],
                                  dtypes.int64)
    default_value = constant_op.constant([-1, -2, -3, -4], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=0,
        deleted_key=-1,
        initial_num_buckets=4,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

    self.evaluate(
        table.insert(
            constant_op.constant([14], dtypes.int64),
            constant_op.constant([[2, 3, 4, 5]], dtypes.int64)))
    self.assertAllEqual(4, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    remove_string = constant_op.constant([12, 16], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    input_string = constant_op.constant([11, 12, 14, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([4, 4],
                        output.shape,
                        msg="Saw shape: %s" % output.shape)

    result = self.evaluate(output)
    self.assertAllEqual(
        [[0, 1, 2, 3], [-1, -2, -3, -4], [2, 3, 4, 5], [-1, -2, -3, -4]],
        result)

  def testVectorKeys(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([[0, 1], [1, 2], [1, 3]], dtypes.int64)
    values = constant_op.constant([10, 11, 12], dtypes.int64)
    empty_key = constant_op.constant([0, 3], dtypes.int64)
    deleted_key = constant_op.constant([-1, -1], dtypes.int64)
    default_value = constant_op.constant(-1, dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        initial_num_buckets=8,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    self.evaluate(
        table.insert(
            constant_op.constant([[0, 0]], dtypes.int64),
            constant_op.constant([13], dtypes.int64)))
    self.assertAllEqual(4, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    remove_string = constant_op.constant([[1, 2], [7, 8]], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    input_string = constant_op.constant([[0, 1], [1, 2], [1, 3], [0, 2]],
                                        dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([4], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([10, -1, 12, -1], result)

  def testResize(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        initial_num_buckets=4,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

    keys2 = constant_op.constant([12, 99], dtypes.int64)
    self.evaluate(table.remove(keys2))
    self.assertAllEqual(2, self.evaluate(table.size()))
    self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

    keys3 = constant_op.constant([13, 14, 15, 16, 17], dtypes.int64)
    values3 = constant_op.constant([3, 4, 5, 6, 7], dtypes.int64)

    self.evaluate(table.insert(keys3, values3))
    self.assertAllEqual(6, self.evaluate(table.size()))
    self.assertAllEqual(16, len(self.evaluate(table.export()[0])))

    keys4 = constant_op.constant([10, 11, 12, 13, 14, 15, 16, 17, 18],
                                 dtypes.int64)
    output = table.lookup(keys4)
    self.assertAllEqual([-1, 0, -1, 3, 4, 5, 6, 7, -1], self.evaluate(output))

  def testExport(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([1, 2, 3, 4], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=100,
        deleted_key=200,
        initial_num_buckets=8,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    keys2 = constant_op.constant([12, 15], dtypes.int64)
    self.evaluate(table.remove(keys2))
    self.assertAllEqual(3, self.evaluate(table.size()))

    exported_keys, exported_values = table.export()

    np_keys = self.evaluate(exported_keys)
    np_values = self.evaluate(exported_values)

    self.assertAllEqual(8, len(np_keys))
    self.assertAllEqual(8, len(np_values))

    # pair up keys and values, drop extra added dimension
    pairs = np.dstack((np_keys.flatten(), np_values.flatten()))[0]
    # sort by key
    pairs = pairs[pairs[:, 0].argsort()]
    self.assertAllEqual([[11, 1], [13, 3], [14, 4], [100, 0], [100, 0],
                         [100, 0], [100, 0], [200, 2]], pairs)

  @test_util.run_v1_only("Saver V1 only")
  def testSaveRestore(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      default_value = -1
      empty_key = 0
      deleted_key = -1
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([12, 15], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([11, 14], dtypes.int64),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, -1, 2, 3], output)

  @test_util.run_v1_only("Saver V1 only")
  def testSaveRestoreOnlyTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      default_value = -1
      empty_key = 0
      deleted_key = -1
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver([table])

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([12, 15], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([11, 14], dtypes.int64),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver([table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, -1, 2, 3], output)

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self, is_anonymous):
    if is_anonymous and not context.executing_eagerly():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    default_value = -1
    empty_key = 0
    deleted_key = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    save_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32,
        experimental_is_anonymous=is_anonymous)

    save_checkpoint = trackable.Checkpoint(table=save_table)

    self.assertAllEqual(0, self.evaluate(save_table.size()))
    self.evaluate(save_table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(save_table.size()))
    self.assertAllEqual(32, len(self.evaluate(save_table.export()[0])))

    save_path = save_checkpoint.save(save_prefix)
    del save_table, save_checkpoint

    load_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=64,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(
        load_table.insert(
            constant_op.constant([11, 14], dtypes.int64),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(load_table.size()))
    self.assertAllEqual(64, len(self.evaluate(load_table.export()[0])))

    restore_checkpoint = trackable.Checkpoint(table=load_table)

    # Restore the saved values in the parameter nodes.
    restore_checkpoint.restore(save_path).run_restore_ops()

    self.assertAllEqual(3, self.evaluate(load_table.size()))
    self.assertAllEqual(32, len(self.evaluate(load_table.export()[0])))

    input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
    output = load_table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    empty_key = 0
    deleted_key = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 0)
    root.table.insert(keys, values)
    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.table.lookup(12), 1)
    self.assertEqual(root.table.lookup(10), -1)
    self.assertEqual(len(root.table.export()[0]), 32)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)

  @test_util.run_v1_only("Saver V1 only")
  def testVectorSaveRestore(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "vector_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-2, -3], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [12, 13], [13, 14]],
                                  dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [2, 4], [4, 5]],
                                    dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([[12, 13], [16, 17]], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-2, -3], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([[21, 22], [23, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2], [4, 5], [-1, -2]],
                          self.evaluate(output))

  @test_util.run_v1_only("Saver V1 only")
  def testVectorScalarSaveRestore(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "vector_scalar_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [12, 13], [13, 14]],
                                  dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([[12, 13], [15, 16]], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([3, 4], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([0, 1, -1, 3, -1], output)

  def testReprobe(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Insert 6 keys into a table with 8 buckets.
    # The values are chosen to make sure collisions occur when using GCC STL
    keys = constant_op.constant([11, 12, 13, 19, 20, 21], dtypes.int64)
    values = constant_op.constant([51, 52, 53, 54, 55, 56], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        initial_num_buckets=8,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(6, self.evaluate(table.size()))

    input_string = constant_op.constant([10, 11, 12, 13, 14, 19, 20, 21, 22],
                                        dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([9], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([-1, 51, 52, 53, -1, 54, 55, 56, -1], result)

  def testCustomEmptyKey(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 0, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=12,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([11, 0, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testErrors(self, is_anonymous):
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)

    # Inserting the empty key returns an error
    keys1 = constant_op.constant([11, 0], dtypes.int64)
    values1 = constant_op.constant([0, 1], dtypes.int64)
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "empty_key"):
      self.evaluate(table.insert(keys1, values1))

    # Looking up the empty key returns an error
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "empty_key"):
      self.evaluate(table.lookup(keys1))

    # Inserting the deleted key returns an error
    keys2 = constant_op.constant([11, -1], dtypes.int64)
    values2 = constant_op.constant([0, 1], dtypes.int64)
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "deleted_key"):
      self.evaluate(table.insert(keys2, values2))

    # Looking up the empty key returns an error
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "deleted_key"):
      self.evaluate(table.lookup(keys2))

    # Arbitrary tensors of keys are not supported
    keys = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
    values = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Expected key shape"):
      self.evaluate(table.lookup(keys))
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Expected key shape"):
      self.evaluate(table.insert(keys, values))

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Number of buckets must be"):
      table2 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=17,
          deleted_key=-1,
          initial_num_buckets=12,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table2.size()))

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "Empty and deleted keys must have same shape"):
      table3 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=42,
          deleted_key=[1, 2],
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table3.size()))

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Empty and deleted keys cannot be equal"):
      table4 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=42,
          deleted_key=42,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table4.size()))

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Empty and deleted keys cannot be equal"):
      table5 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=[1, 2, 3],
          deleted_key=[1, 2, 3],
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table5.size()))

  @test_util.run_in_graph_and_eager_modes
  def testStringToResource(self, is_anonymous):
    v = variables.Variable(1.)
    v1 = variables.Variable(1.)
    table = lookup_ops.DenseHashTable(
        dtypes.string,
        dtypes.resource,
        default_value=v.handle,
        empty_key="<empty>",
        deleted_key="<deleted>",
        experimental_is_anonymous=is_anonymous)
    self.assertEqual([], table.lookup("not_found").shape)
    table.insert("v1", v1.handle)
    self.assertEqual([], table.lookup("v1").shape)

  def testExportShapeInference(self, is_anonymous):
    default_value = -1
    empty_key = 0
    deleted_key = -1
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))


class IndexTableFromFile(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_string_index_table_from_file(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")

    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_multicolumn_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain\t300", "salad\t20", "surgery\t1"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_column_index=0,
        value_column_index=lookup_ops.TextFileIndex.LINE_NUMBER)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_multicolumn_file_custom_delimiter(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain 300", "salad 20", "surgery 1"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_column_index=0,
        value_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        delimiter=" ")
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_file_tensor_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    vocabulary_file = constant_op.constant(vocabulary_file)
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))
    if not context.executing_eagerly():
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  @test_util.run_v1_only("placeholder usage")
  def test_string_index_table_from_file_placeholder_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_placeholder = array_ops.placeholder(dtypes.string, [])
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_placeholder, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)

      feed_dict = {vocabulary_placeholder.name: vocabulary_file}
      lookup_ops.tables_initializer().run(feed_dict=feed_dict)
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))
      self.assertEqual(0,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_int32_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab2.txt", values=("42", "1", "-1000"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_dtype=dtypes.int32)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int64_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab3.txt", values=("42", "1", "-1000"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_dtype=dtypes.int64)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_index_table_from_file_with_default_value(self):
    default_value = -42
    vocabulary_file = self._createVocabFile("f2i_vocab4.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, default_value=default_value)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, default_value), self.evaluate(ids))

  def test_index_table_from_file_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab5.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1000)
    ids = table.lookup(
        constant_op.constant(["salad", "surgery", "tarkus", "toccata"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual(
        (
            1,  # From vocabulary file.
            2,  # From vocabulary file.
            867,  # 3 + fingerprint("tarkus") mod 300.
            860),  # 3 + fingerprint("toccata") mod 300.
        self.evaluate(ids))

  def test_index_table_from_file_fails_with_empty_vocabulary_file_name(self):
    self.assertRaises(
        ValueError, lookup_ops.index_table_from_file, vocabulary_file="")

  def test_index_table_from_file_fails_with_empty_vocabulary(self):
    self.assertRaises(
        ValueError, lookup_ops.index_table_from_file, vocabulary_file=None)

  def test_index_table_from_file_str_fails_with_zero_size_vocabulary(self):
    vocabulary_file = self._createVocabFile("zero_vocab_str.txt")
    self.assertRaisesRegex(
        ValueError, "`vocab_size` must be greater than 0, got 0 for "
        "vocabulary_file: .*zero_vocab_str.txt",
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

  def test_index_table_from_file_tensor_fails_with_zero_size_vocabulary(self):
    vocabulary_file = constant_op.constant(
        self._createVocabFile("zero_vocab_tensor.txt"))
    self.assertRaisesRegex(
        ValueError, "`vocab_size` must be greater than 0, got 0 for "
        "vocabulary_file: .*zero_vocab_tensor.txt",
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

  def test_index_table_from_file_with_vocab_size_too_small(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, vocab_size=2)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, -1, -1), self.evaluate(ids))
    self.assertEqual(2, self.evaluate(table.size()))

  def test_index_table_from_file_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Invalid vocab_size"):
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.evaluate(table.initializer)

  def test_index_table_from_file_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab8.txt")

    self.assertRaises(
        ValueError,
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, vocab_size=3)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, -1), self.evaluate(ids))
    self.assertEqual(3, self.evaluate(table.size()))

  def test_index_table_from_file_with_invalid_hashers(self):
    vocabulary_file = self._createVocabFile("invalid_hasher.txt")
    with self.assertRaises(TypeError):
      lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=3,
          num_oov_buckets=1,
          hasher_spec=1)

    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        vocab_size=3,
        num_oov_buckets=1,
        hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

    self.assertRaises(ValueError, table.lookup,
                      constant_op.constant(["salad", "surgery", "tarkus"]))

  def test_index_table_from_file_table_ref_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab9.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1)
    self.assertIsNotNone(table.resource_handle)

  def test_index_table_from_file_table_ref_without_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab10.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=0)
    self.assertIsNotNone(table.resource_handle)


class IndexTableFromTensor(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_index_table_from_tensor_with_tensor_init(self):
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=("brain", "salad", "surgery"), num_oov_buckets=1)

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(
            table.lookup(constant_op.constant(("salad", "surgery", "tarkus"))))
    else:
      # Reinitializing a table in eager should work.
      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=("brain", "salad", "surgery"), num_oov_buckets=1)
    self.evaluate(lookup_ops.tables_initializer())
    ids = table.lookup(constant_op.constant(("salad", "surgery", "tarkus")))
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int32_index_table_from_tensor_with_tensor_init(self):
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int32)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int64_index_table_from_tensor_with_tensor_init(self):
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int64)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_index_table_from_tensor_with_default_value(self):
    default_value = -42
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=["brain", "salad", "surgery"],
        default_value=default_value)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, default_value), self.evaluate(ids))

  def test_index_table_from_tensor_missing_vocabulary_list(self):
    with self.assertRaisesRegex(ValueError,
                                "`vocabulary_list` must be specified"):
      lookup_ops.index_table_from_tensor(
          vocabulary_list=None, num_oov_buckets=1)

  def test_index_table_from_tensor_empty_vocabulary_list(self):
    with self.assertRaisesRegex(errors_impl.OpError,
                                "keys and values cannot be empty"):
      _ = lookup_ops.index_table_from_tensor(
          vocabulary_list=np.array([], dtype=np.str_), num_oov_buckets=1)
      self.evaluate(lookup_ops.tables_initializer())

  def test_index_table_from_tensor_with_invalid_hashers(self):
    with self.assertRaises(TypeError):
      lookup_ops.index_table_from_tensor(
          vocabulary_list=["brain", "salad", "surgery"],
          num_oov_buckets=1,
          hasher_spec=1)

    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=["brain", "salad", "surgery"],
        num_oov_buckets=1,
        hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

    self.assertRaises(ValueError, table.lookup,
                      constant_op.constant(["salad", "surgery", "tarkus"]))


class IndexToStringTableFromFileTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_index_to_string_table(self):
    vocabulary_path = self._createVocabFile("i2f_vocab1.txt")
    # vocabulary_file supports string and tensor
    type_funcs = [str, constant_op.constant]
    for type_func in type_funcs:
      vocabulary_file = type_func(vocabulary_path)
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file)
      features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          self.evaluate(features))

  def test_index_to_string_table_from_multicolumn_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain\t300", "salad\t20", "surgery\t1"))
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file,
        key_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_column_index=0)
    features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                        self.evaluate(features))

  def test_index_to_string_table_from_multicolumn_file_custom_delimiter(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain 300", "salad 20", "surgery 1"))
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file,
        key_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_column_index=0,
        delimiter=" ")
    features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                        self.evaluate(features))

  def test_index_to_string_table_with_default_value(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file, default_value=default_value)
    features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", b"surgery", default_value),
                        self.evaluate(features))

  def test_index_to_string_table_with_vocab_size_too_small(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file,
        vocab_size=2,
        default_value=default_value)
    features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", default_value, default_value),
                        self.evaluate(features))

  def test_index_to_string_table_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Invalid vocab_size"):
      _ = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.evaluate(lookup_ops.tables_initializer())

  def test_index_to_string_table_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file, vocab_size=3)
    features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", b"surgery", b"UNK"), self.evaluate(features))


class IndexToStringTableFromTensorTest(test.TestCase):

  def test_index_to_string_table_from_tensor(self):
    vocabulary_list = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=vocabulary_list)

    indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    features = table.lookup(indices)
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                        self.evaluate(features))

  def test_duplicate_entries(self):
    vocabulary_list = constant_op.constant(["hello", "hello"])
    table = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=vocabulary_list)
    indices = constant_op.constant([0, 1, 4], dtypes.int64)
    features = table.lookup(indices)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"hello", b"hello", b"UNK"), self.evaluate(features))

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    vocabulary_list = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=vocabulary_list, default_value=default_value)
    indices = constant_op.constant([1, 2, 4], dtypes.int64)
    features = table.lookup(indices)
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", b"surgery", default_value),
                        self.evaluate(features))


class IdTableWithHashBucketsTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testStringIdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value),
        oov_buckets)

    self.evaluate(table.initializer)

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out = table.lookup(input_string)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt32IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
            default_value),
        oov_buckets,
        key_dtype=dtypes.int32)

    self.evaluate(table.initializer)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt64IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
            default_value), oov_buckets)

    self.evaluate(table.initializer)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testStringIdTableWithOnlyHashBucket(self):
    oov_buckets = 5

    # Set a table that only uses hash buckets, for each input value returns
    # an id calculated by fingerprint("input") mod oov_buckets.
    table = lookup_ops.IdTableWithHashBuckets(None, oov_buckets)
    self.evaluate(table.initializer)

    values = constant_op.constant(("brain", "salad", "surgery"))

    out = table.lookup(values)
    self.assertAllEqual(
        [
            3,  # fingerprint("brain") mod 5.
            1,  # fingerprint("salad") mod 5.
            4  # fingerprint("surgery") mod 5
        ],
        self.evaluate(out))
    self.assertEqual(oov_buckets, self.evaluate(table.size()))

  def testInt32IdTableWithOnlyHashBucket(self):
    oov_buckets = 5

    # Set a table that only uses hash buckets, for each input value returns
    # an id calculated by fingerprint("input") mod oov_buckets.
    table = lookup_ops.IdTableWithHashBuckets(
        None, oov_buckets, key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    input_string = constant_op.constant([42, 1, -1000], dtype=dtypes.int32)

    out = table.lookup(input_string)
    self.assertAllEqual(
        [
            1,  # fingerprint("42") mod 5.
            4,  # fingerprint("1") mod 5.
            2  # fingerprint("-1000") mod 5
        ],
        self.evaluate(out))
    self.assertEqual(oov_buckets, self.evaluate(table.size()))

  def testFloat64IdTableWithOnlyHashBucket(self):
    with self.assertRaisesRegex(TypeError, "Invalid `key_dtype`"):
      lookup_ops.IdTableWithHashBuckets(
          None, num_oov_buckets=5, key_dtype=dtypes.float64)

  def testBoolIdTableWithOnlyHashBucket(self):
    with self.assertRaisesRegex(TypeError, "Invalid `key_dtype`"):
      lookup_ops.IdTableWithHashBuckets(
          None, num_oov_buckets=5, key_dtype=dtypes.bool)

  def testIdTableWithHashBucketsWithMultipleInitializers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 3

    vocab_table = lookup_ops.StaticHashTable(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size), default_value)
    table1 = lookup_ops.IdTableWithHashBuckets(
        vocab_table,
        oov_buckets,
        hasher_spec=lookup_ops.FastHashSpec,
        name="table1")

    table2 = lookup_ops.IdTableWithHashBuckets(
        vocab_table,
        oov_buckets,
        hasher_spec=lookup_ops.StrongHashSpec((1, 2)),
        name="table2")

    self.evaluate(lookup_ops.tables_initializer())

    input_string = constant_op.constant(
        ["fruit", "brain", "salad", "surgery", "UNK"])

    out1 = table1.lookup(input_string)
    out2 = table2.lookup(input_string)

    out1, out2 = self.evaluate([out1, out2])
    self.assertAllEqual([5, 0, 1, 2, 5], out1)
    self.assertAllEqual([5, 0, 1, 2, 3], out2)
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))
    if not context.executing_eagerly():
      test_util.assert_ops_in_graph({
          "table1_Lookup/hash_bucket": "StringToHashBucketFast",
          "table2_Lookup/hash_bucket": "StringToHashBucketStrong",
      }, ops.get_default_graph())

  def testIdTableWithHashBucketsInitializationAcrossSessions(self):
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table1 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value), oov_buckets)

    self.evaluate(table1.initializer)

    input_string_1 = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out1 = table1.lookup(input_string_1)

    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out1))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))

    default_value = -1
    vocab_size = 3
    oov_buckets = 1

    # Underlying lookup table already initialized in previous session.
    # No need to call self.evaluate(table2.initializer)
    table2 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value), oov_buckets)

    input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

    out2 = table2.lookup(input_string_2)

    self.assertAllEqual([3, 1, 3], self.evaluate(out2))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testIdTableWithHashBucketsWithMultipleInitializersDifferentDefault(self):
    vocab_file = self._createVocabFile("feat_to_id_6.txt")
    default_value1 = -1
    vocab_size = 3
    oov_buckets = 0
    table1 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value1),
        oov_buckets)

    default_value2 = -2
    table2 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value2),
        oov_buckets)

    self.evaluate(lookup_ops.tables_initializer())

    input_string_1 = constant_op.constant(
        ["brain", "salad", "surgery", "UNK"])
    input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

    out1 = table1.lookup(input_string_1)
    out2 = table2.lookup(input_string_2)

    out1, out2 = self.evaluate([out1, out2])
    self.assertAllEqual([0, 1, 2, -1], out1)
    self.assertAllEqual([-2, 1, -2], out2)
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testSparseTensor(self):
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
            -1), 1)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testRaggedTensor(self):
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
            -1), 1)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)
    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt32SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32RaggedTensor(self):
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_row_splits, dtypes.int32))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt64SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int64)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64RaggedTensor(self):
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int64)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testIdTableWithHashBucketsWithInvalidHashers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    lookup_table = lookup_ops.StaticHashTable(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size), default_value)

    with self.assertRaises(TypeError):
      lookup_ops.IdTableWithHashBuckets(
          lookup_table, oov_buckets, hasher_spec=1)

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_table,
        oov_buckets,
        hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    with self.assertRaises(ValueError):
      table.lookup(input_string)

    with self.assertRaises(ValueError):
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table, oov_buckets, hasher_spec=lookup_ops.StrongHashSpec([]))

    with self.assertRaises(ValueError):
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup_ops.StrongHashSpec([1, 2, 3]))

    with self.assertRaises(TypeError):
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup_ops.StrongHashSpec([None, 2]))

  def testIdTableWithHashBucketsNoInnerTable(self):
    table = lookup_ops.IdTableWithHashBuckets(None, num_oov_buckets=1)
    self.assertIsNone(table.resource_handle)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class MutableHashTableOpTest(test.TestCase):

  def testMutableHashTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["tarkus", "tank"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

    exported_keys, exported_values = table.export()

    # exported data is in the order of the internal map, i.e. undefined
    sorted_keys = np.sort(self.evaluate(exported_keys))
    sorted_values = np.sort(self.evaluate(exported_values))
    self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
    self.assertAllEqual([0, 1, 2], sorted_values)

  # TODO(https://github.com/tensorflow/tensorflow/issues/24439): remove exepectedFailure when fixed
  @unittest.expectedFailure
  @test_util.run_v2_only
  def testImportedHashTable(self, is_anonymous):
    g = ops.Graph()
    with g.as_default():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.evaluate(table.insert(keys, values))
      op = table.lookup(constant_op.constant(["brain", "salad", "tank"]))
      meta_graph = saver.export_meta_graph()

    def f():
      saver.import_meta_graph(meta_graph)
      return ops.get_default_graph().get_tensor_by_name(op.name)

    wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual([0, 1, -1], wrapped())

  @test_util.run_v1_only("SaverV1")
  def testSaveRestore(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant(["b", "c", "d"], dtypes.string)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = -1
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)
      self.evaluate(
          table.insert(
              constant_op.constant(["a", "c"], dtypes.string),
              constant_op.constant([12, 24], dtypes.int64)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                          dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v1_only("SaverV1")
  def testSaveRestoreOnlyTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant(["b", "c", "d"], dtypes.string)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver([table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      default_val = -1
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)
      self.evaluate(
          table.insert(
              constant_op.constant(["a", "c"], dtypes.string),
              constant_op.constant([12, 24], dtypes.int64)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver([table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                          dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self, is_anonymous):
    if is_anonymous and not context.executing_eagerly():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")

    default_val = -1
    keys = constant_op.constant(["b", "c", "d"], dtypes.string)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        name="t1",
        checkpoint=True,
        experimental_is_anonymous=is_anonymous)

    checkpoint = trackable.Checkpoint(table=table, v0=v0, v1=v1)
    self.evaluate([v0.initializer, v1.initializer])

    # Check that the parameter nodes have been initialized.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(0, self.evaluate(table.size()))
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    save_path = checkpoint.save(save_prefix)
    del table, checkpoint, v0, v1

    v0 = variables.Variable(-1.0, name="v0")
    v1 = variables.Variable(-1.0, name="v1")
    default_val = -1
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        name="t1",
        checkpoint=True,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(
        table.insert(
            constant_op.constant(["a", "c"], dtypes.string),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(table.size()))

    checkpoint = trackable.Checkpoint(table=table, v0=v0, v1=v1)

    # Restore the saved values in the parameter nodes.
    checkpoint.restore(save_path).run_restore_ops()
    # Check that the parameter nodes have been restored.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                        dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 0)
    root.table.insert(keys, values)
    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.table.lookup(12), 1)
    self.assertEqual(root.table.lookup(10), -1)
    self.assertEqual(len(root.table.export()[0]), 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)

  @test_util.run_v1_only("Multiple sessions")
  def testSharing(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.string,
        "-",
        name="t1",
        experimental_is_anonymous=is_anonymous)

    # Populate the table in the first session
    with session1:
      self.assertAllEqual(0, table.size())

      keys = constant_op.constant([11, 12], dtypes.int64)
      values = constant_op.constant(["a", "b"])
      table.insert(keys, values).run()
      self.assertAllEqual(2, table.size())

      output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
      self.assertAllEqual([b"a", b"b", b"-"], output)

    # Verify that we can access the shared data from the second session
    with session2:
      self.assertAllEqual(2, table.size())

      output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
      self.assertAllEqual([b"-", b"a", b"b"], output)

  def testMutableHashTableOfTensors(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
    values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                  dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["tarkus", "tank"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [2, 3], [-1, -1]], result)

    exported_keys, exported_values = table.export()
    # exported data is in the order of the internal map, i.e. undefined
    sorted_keys = np.sort(self.evaluate(exported_keys))
    sorted_values = np.sort(self.evaluate(exported_values), axis=0)
    self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
    sorted_expected_values = np.sort([[4, 5], [2, 3], [0, 1]], axis=0)
    self.assertAllEqual(sorted_expected_values, sorted_values)

  def testMutableHashTableExportInsert(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
    table1 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table1.size()))
    self.evaluate(table1.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table1.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    expected_output = [[0, 1], [2, 3], [-1, -1]]
    output1 = table1.lookup(input_string)
    self.assertAllEqual(expected_output, self.evaluate(output1))

    exported_keys, exported_values = table1.export()
    self.assertAllEqual(3, self.evaluate(exported_keys).size)
    self.assertAllEqual(6, self.evaluate(exported_values).size)

    # Populate a second table from the exported data
    table2 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table2.size()))
    self.evaluate(table2.insert(exported_keys, exported_values))
    self.assertAllEqual(3, self.evaluate(table2.size()))

    # Verify lookup result is still the same
    output2 = table2.lookup(input_string)
    self.assertAllEqual(expected_output, self.evaluate(output2))

  def testMutableHashTableOfTensorsInvalidShape(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    # Shape [6] instead of [3, 2]
    values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Shape [2,3] instead of [3, 2]
    values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Shape [2, 2] instead of [3, 2]
    values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Shape [3, 1] instead of [3, 2]
    values = constant_op.constant([[0], [2], [4]], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Valid Insert
    values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

  def testMutableHashTableInvalidDefaultValue(self, is_anonymous):
    default_val = constant_op.constant([[-1, -1]], dtypes.int64)
    with self.assertRaisesOpError("Default value must be a vector"):
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table.size()))

  def testMutableHashTableDuplicateInsert(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery", "brain"])
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([3, 1, -1], result)

  def testMutableHashTableFindHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)
    self.assertAllEqual([2, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testMutableHashTableFindWithInvalidShapeDefaultValue(self, is_anonymous):
    default_val = [-1, -1]
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    input_string = constant_op.constant([["brain", "salad"], ["tank",
                                                              "tarkus"]])

    invalid_default_val = constant_op.constant(
        [[-2, -3], [-4, -5], [-6, -7], [-8, -9]], dtypes.int64)

    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "Expected shape \[2\] or \[2,2,2\] for default value, got \[4,2]"):
      self.evaluate(table.lookup(input_string, invalid_default_val))

    invalid_default_val = constant_op.constant([[[-2, -3], [-4, -5]]],
                                               dtypes.int64)
    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "Expected shape \[2\] or \[2,2,2\] for default value, got \[1,2,2\]"):
      self.evaluate(table.lookup(input_string, invalid_default_val))

  def testMutableHashTableFindHighRankScalarWithDynamicDefaultValue(
      self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"], ["tank",
                                                              "tarkus"]])

    dynamic_default_val = constant_op.constant([[-2, -3], [-4, -5]],
                                               dtypes.int64)
    output = table.lookup(input_string, dynamic_default_val)
    self.assertAllEqual([2, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-4, -5]], result)

  def testMutableHashTableFindHighRankVectorWithDynamicDefaultValue(
      self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = [-1, -1]
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"], ["tank",
                                                              "tarkus"]])

    dynamic_default_val = constant_op.constant(
        [[[-2, -3], [-4, -5]], [[-6, -7], [-8, -9]]], dtypes.int64)
    output = table.lookup(input_string, dynamic_default_val)
    self.assertAllEqual([2, 2, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[[0, 1], [2, 3]], [[-6, -7], [-8, -9]]], result)

  def testMutableHashTableInsertHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
    values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, 3, -1], result)

  def testMutableHashTableRemoveHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
    values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["salad", "tarkus"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, -1, 3, -1], result)

  def testMutableHashTableOfTensorsFindHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                  dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)
    self.assertAllEqual([2, 2, 3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual(
        [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def testMutableHashTableOfTensorsRemoveHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                  dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    remove_string = constant_op.constant([["brain", "tank"]])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(2, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["surgery", "tank"]])
    output = table.lookup(input_string)
    self.assertAllEqual([2, 2, 3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual(
        [[[-1, -1, -1], [2, 3, 4]], [[4, 5, 6], [-1, -1, -1]]], result)

  def testMultipleMutableHashTables(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)

    table1 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    table2 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    table3 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(table1.insert(keys, values))
    self.evaluate(table2.insert(keys, values))
    self.evaluate(table3.insert(keys, values))

    self.assertAllEqual(3, self.evaluate(table1.size()))
    self.assertAllEqual(3, self.evaluate(table2.size()))
    self.assertAllEqual(3, self.evaluate(table3.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output1 = table1.lookup(input_string)
    output2 = table2.lookup(input_string)
    output3 = table3.lookup(input_string)

    out1, out2, out3 = self.evaluate([output1, output2, output3])
    self.assertAllEqual([0, 1, -1], out1)
    self.assertAllEqual([0, 1, -1], out2)
    self.assertAllEqual([0, 1, -1], out3)

  def testMutableHashTableWithTensorDefault(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testSignatureMismatch(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    # insert with keys of the wrong type
    with self.assertRaises(ValueError):
      self.evaluate(table.insert(constant_op.constant([4, 5, 6]), values))

    # insert with values of the wrong type
    with self.assertRaises(ValueError):
      self.evaluate(table.insert(keys, constant_op.constant(["a", "b", "c"])))

    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string_ref = variables.Variable("brain")
    input_int64_ref = variables.Variable(-1, dtype=dtypes.int64)
    self.evaluate(variables.global_variables_initializer())

    # Ref types do not produce an insert signature mismatch.
    self.evaluate(table.insert(input_string_ref, input_int64_ref))
    self.assertAllEqual(3, self.evaluate(table.size()))

    # Ref types do not produce a lookup signature mismatch.
    self.assertEqual(-1, self.evaluate(table.lookup(input_string_ref)))

    # lookup with keys of the wrong type
    input_string = constant_op.constant([1, 2, 3], dtypes.int64)
    with self.assertRaises(ValueError):
      self.evaluate(table.lookup(input_string))

    # default value of the wrong type
    with self.assertRaises(TypeError):
      lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          "UNK",
          experimental_is_anonymous=is_anonymous)

  def testMutableHashTableStringFloat(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1.5
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1.1, 2.2], dtypes.float32)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.float32,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllClose([0, 1.1, default_val], result)

  def testMutableHashTableIntFloat(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1.0
    keys = constant_op.constant([3, 7, 0], dtypes.int64)
    values = constant_op.constant([7.5, -1.2, 9.9], dtypes.float32)
    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.float32,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([7, 0, 11], dtypes.int64)
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllClose([-1.2, 9.9, default_val], result)

  def testMutableHashTableInt64String(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = "n/a"
    keys = constant_op.constant([0, 1, 2], dtypes.int64)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.string,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([0, 1, 3], dtypes.int64)
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual((b"brain", b"salad", b"n/a"), result)

  def testExportShapeInference(self, is_anonymous):
    default_value = -1
    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))


class MutableHashTableBenchmark(test.Benchmark):

  def _create_table(self):
    return lookup_ops.MutableHashTable(dtypes.int64, dtypes.float32, 0.0)

  def benchmark_single_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable(1.0)
    insert = table.insert(0, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) == 1

  def benchmark_many_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable(1.0)
    insert = table.insert(c, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) >= 10000

  def benchmark_single_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) == 32

  def benchmark_many_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(32 * c + list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) >= 1000 * 32


class DenseHashTableBenchmark(MutableHashTableBenchmark):

  def _create_table(self):
    return lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.float32,
        default_value=0.0,
        empty_key=-1,
        deleted_key=-2)


if __name__ == "__main__":
  test.main()
