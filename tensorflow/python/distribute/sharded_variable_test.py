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
"""Tests for ShardedVariable."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.test_util import get_cluster_def
from tensorflow.python.distribute.test_util import TestClusterParams
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest

# We create one cluster to share between tests. The cluster should be large
# enough to accommodate all the tests. Adjust the following constants as needed
# but be aware of resource limitations in OSS tests.
test_cluster_params = TestClusterParams(None, 2, 3)


def _load_and_run(
    model_dir,
    inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  """Load a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    meta_graph_def = loader.load(session, [tag_constants.SERVING], model_dir)
    signature = meta_graph_def.signature_def[signature_key]
    feed_dict = {}
    for arg_name in inputs.keys():
      input_tensor = session.graph.get_tensor_by_name(
          signature.inputs[arg_name].name)
      feed_dict[input_tensor] = inputs[arg_name]
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = session.graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


class PartitionerTest(test.TestCase):

  def test_fixed_shards_partitioner(self):
    partitioner = sharded_variable.FixedShardsPartitioner(num_shards=2)
    got = partitioner(tensor_shape.TensorShape([10, 3]), dtypes.float32)
    self.assertAllEqual(got, [2, 1])

  def test_min_size_partitioner(self):
    partitioner = sharded_variable.MinSizePartitioner(
        min_shard_bytes=4, max_shards=2)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [2, 1])

    partitioner = sharded_variable.MinSizePartitioner(
        min_shard_bytes=4, max_shards=10)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [6, 1])

  def test_max_size_partitioner(self):
    partitioner = sharded_variable.MaxSizePartitioner(max_shard_bytes=4)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [6, 1])

    partitioner = sharded_variable.MaxSizePartitioner(
        max_shard_bytes=4, max_shards=2)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [2, 1])

    partitioner = sharded_variable.MaxSizePartitioner(max_shard_bytes=1024)
    got = partitioner(tensor_shape.TensorShape([6, 1]), dtypes.float32)
    self.assertAllEqual(got, [1, 1])


class ShardedVariableTest(test.TestCase, parameterized.TestCase):

  def test_sharded_variable_simple(self):
    v0 = variables_lib.Variable([0])
    v1 = variables_lib.Variable([1])
    s = sharded_variable.ShardedVariable([v0, v1], name='s')
    self.assertEqual(s.variables[0], v0)
    self.assertEqual(s.variables[1], v1)
    self.assertEqual(s.shape.as_list(), [2])
    self.assertEqual(s.dtype, v0.dtype)
    self.assertEqual(s.name, 's')

  def test_assign(self):
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    ret = s.assign([[4, 4], [5, 5], [6, 6], [7, 7]])
    self.assertAllEqual(self.evaluate(s.variables[0]), [[4, 4]])
    self.assertAllEqual(self.evaluate(s.variables[1]), [[5, 5], [6, 6]])
    self.assertAllEqual(self.evaluate(s.variables[2]), [[7, 7]])
    self.assertIs(ret, s)

  def test_assign_add(self):
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    ret = s.assign_add([[1, 1], [1, 1], [2, 2], [2, 2]])
    self.assertAllEqual(self.evaluate(s.variables[0]), [[1, 1]])
    self.assertAllEqual(self.evaluate(s.variables[1]), [[2, 2], [4, 4]])
    self.assertAllEqual(self.evaluate(s.variables[2]), [[5, 5]])
    self.assertIs(ret, s)

  def test_assign_sub(self):
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    ret = s.assign_sub([[0, 0], [1, 1], [1, 1], [3, 3]])
    self.assertAllEqual(self.evaluate(s.variables[0]), [[0, 0]])
    self.assertAllEqual(self.evaluate(s.variables[1]), [[0, 0], [1, 1]])
    self.assertAllEqual(self.evaluate(s.variables[2]), [[0, 0]])
    self.assertIs(ret, s)

  def test_scatter_add_uneven_partition(self):
    v = variables_lib.Variable(array_ops.zeros((32, 1)))
    sparse_delta = indexed_slices.IndexedSlices(
        values=constant_op.constant([[0.], [1.], [2.], [3.], [4.], [5.]]),
        indices=constant_op.constant([0, 10, 11, 12, 30, 31]))

    v0 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    v.scatter_add(sparse_delta)
    sv.scatter_add(sparse_delta)
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

    @def_function.function
    def func():
      v.scatter_add(sparse_delta)
      sv.scatter_add(sparse_delta)

    func()
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

  @parameterized.parameters('scatter_add', 'scatter_div', 'scatter_max',
                            'scatter_min', 'scatter_mul', 'scatter_sub',
                            'scatter_update')
  def test_scatter_ops_even_partition(self, op):
    v = variables_lib.Variable(array_ops.zeros((30, 1)))
    # Make sure values does not contain 0 due to testing `scatter_div`!
    sparse_delta = indexed_slices.IndexedSlices(
        values=constant_op.constant([[1.], [2.], [3.], [4.], [5.]]),
        indices=constant_op.constant([0, 10, 12, 21, 22]))

    v0 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    getattr(v, op)(sparse_delta, name='scatter_v')
    getattr(sv, op)(sparse_delta, name='scatter_sv')
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

    @def_function.function
    def func():
      getattr(v, op)(sparse_delta, name='scatter_v')
      getattr(sv, op)(sparse_delta, name='scatter_sv')

    func()
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

  def test_batch_scatter_update(self):
    v = variables_lib.Variable(array_ops.zeros((32, 1)))
    sparse_delta = indexed_slices.IndexedSlices(
        values=constant_op.constant([[0.], [1.], [2.], [3.], [4.], [5.]]),
        indices=constant_op.constant([10, 11, 12, 13, 14, 15]))

    v0 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((11, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    v.batch_scatter_update(sparse_delta)
    sv.batch_scatter_update(sparse_delta)
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

    @def_function.function
    def func():
      v.batch_scatter_update(sparse_delta)
      sv.batch_scatter_update(sparse_delta)

    func()
    self.assertAllEqual(v, ops.convert_to_tensor(sv))

  def test_sparse_read(self):
    v = variables_lib.Variable(array_ops.zeros((30, 1)))
    indices = constant_op.constant([0, 10, 12, 21, 22])

    v0 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v1 = variables_lib.Variable(array_ops.zeros((10, 1)))
    v2 = variables_lib.Variable(array_ops.zeros((10, 1)))
    sv = sharded_variable.ShardedVariable([v0, v1, v2])

    self.assertAllEqual(v.sparse_read(indices), sv.sparse_read(indices))

    @def_function.function
    def func():
      return v.sparse_read(indices), sv.sparse_read(indices)

    got, expect = func()
    self.assertAllEqual(got, expect)

  def test_control_dep_on_assign(self):
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])

    @def_function.function
    def func():
      ret = s.assign([[4, 4], [5, 5], [6, 6], [7, 7]])
      with ops.control_dependencies([ret]):
        a = array_ops.ones((1, 1))
      with ops.control_dependencies([control_flow_ops.group(ret)]):
        b = array_ops.ones((1, 1))
      return a, b

    func()

  def test_convert_to_tensor(self):
    v0 = variables_lib.Variable([[0, 0]])
    v1 = variables_lib.Variable([[1, 1], [2, 2]])
    v2 = variables_lib.Variable([[3, 3]])
    s = sharded_variable.ShardedVariable([v0, v1, v2])
    t = ops.convert_to_tensor(s)
    self.assertAllEqual(t, [[0, 0], [1, 1], [2, 2], [3, 3]])

  def test_save_restore(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])
    cp.write(fname)

    self.evaluate(cp.s.variables[0].assign([4]))
    self.assertEqual(self.evaluate(cp.s.variables[0]), [4])

    cp.restore(fname)
    # Tests that the original weights are restored.
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])

  def test_save_restore_different_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [variables_lib.Variable([0, 0, 0, 0])]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')

    # Restore from 4 partitions into 1.
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1, 2, 3])

    self.evaluate(cp2.s.variables[0].assign([5, 10, 15, 20]))
    cp2.write(fname)

    # Restore 1 partition into 4.
    cp.restore(fname)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [5])
    self.assertEqual(self.evaluate(cp.s.variables[1]), [10])
    self.assertEqual(self.evaluate(cp.s.variables[2]), [15])
    self.assertEqual(self.evaluate(cp.s.variables[3]), [20])

  def test_save_restore_4_to_2_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')
    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [
        variables_lib.Variable([0, 0]),
        variables_lib.Variable([0, 0])
    ]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    # Assert that weights from the 4 partitions were loaded here.
    self.assertLen(cp2.s.variables, 2)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1])
    self.assertAllEqual(self.evaluate(cp2.s.variables[1]), [2, 3])

  def test_delayed_restore(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    model = tracking.AutoTrackable()
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    model.s = sharded_variable.ShardedVariable(variables)
    cp = util.Checkpoint(model=model)
    cp.write(fname)

    model2 = tracking.AutoTrackable()
    cp2 = util.Checkpoint(model=model2)
    cp2.restore(fname)
    variables2 = [
        variables_lib.Variable([0]),
        variables_lib.Variable([0]),
        variables_lib.Variable([0]),
        variables_lib.Variable([0])
    ]
    model2.s = sharded_variable.ShardedVariable(variables2)
    self.assertAllEqual(self.evaluate(model2.s.variables[0]), [0])
    self.assertAllEqual(self.evaluate(model2.s.variables[1]), [1])
    self.assertAllEqual(self.evaluate(model2.s.variables[2]), [2])
    self.assertAllEqual(self.evaluate(model2.s.variables[3]), [3])

  def test_delayed_restore_4_to_2_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    model = tracking.AutoTrackable()
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    model.s = sharded_variable.ShardedVariable(variables)
    cp = util.Checkpoint(model=model)
    cp.write(fname)

    model2 = tracking.AutoTrackable()
    cp2 = util.Checkpoint(model=model2)
    cp2.restore(fname)
    variables2 = [
        variables_lib.Variable([0, 0]),
        variables_lib.Variable([0, 0])
    ]
    model2.s = sharded_variable.ShardedVariable(variables2)
    self.assertAllEqual(self.evaluate(model2.s.variables[0]), [0, 1])
    self.assertAllEqual(self.evaluate(model2.s.variables[1]), [2, 3])

  def test_save_graph_def(self):
    root = tracking.AutoTrackable()
    v1 = variables_lib.Variable([3.])
    v2 = variables_lib.Variable([2.])
    root.v = sharded_variable.ShardedVariable([v1, v2])
    root.train = def_function.function(
        lambda x: embedding_ops.embedding_lookup_v2(root.v.variables, x))
    # TODO(b/144057383): Remove the necessity of root.serve once saving context
    # is made to tf.function cache.
    root.serve = def_function.function(
        lambda x: embedding_ops.embedding_lookup_v2(root.v.variables[0], x),
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32, name='x')])

    # Trace and use root.train
    self.assertAllEqual([3., 2.], root.train([0, 1]).numpy())

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, root.serve)
    self.assertAllEqual([3., 2.],
                        _load_and_run(save_dir, {'x': [0, 1]})['output_0'])

    # Continue using root.train for training
    self.assertAllEqual([3., 2.], root.train([0, 1]).numpy())

  def test_validation_errors(self):
    with self.assertRaisesRegex(TypeError, 'should be a non-empty list of'):
      sharded_variable.ShardedVariable(None)

    with self.assertRaisesRegex(TypeError, 'should be a non-empty list of'):
      sharded_variable.ShardedVariable(
          [variables_lib.Variable([0]), 'not-a-variable'])

    with self.assertRaisesRegex(TypeError, 'should be a non-empty list of'):
      sharded_variable.ShardedVariable([])

    with self.assertRaisesRegex(ValueError, 'must have the same dtype'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable([0], dtype='int64'),
          variables_lib.Variable([1], dtype='int32')
      ])

    with self.assertRaisesRegex(ValueError, 'the same shapes except'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable(array_ops.ones((5, 10))),
          variables_lib.Variable(array_ops.ones((5, 20)))
      ])

    with self.assertRaisesRegex(ValueError, '`SaveSliceInfo` should not'):
      v = variables_lib.Variable([0])
      v._set_save_slice_info(
          variables_lib.Variable.SaveSliceInfo(
              full_name='s', full_shape=[2], var_offset=[0], var_shape=[1]))
      sharded_variable.ShardedVariable([v])

  def test_as_function_input(self):
    variables1 = [
        variables_lib.Variable([1]),
        variables_lib.Variable([1]),
    ]
    s = sharded_variable.ShardedVariable(variables1)
    variables2 = [
        variables_lib.Variable([2]),
        variables_lib.Variable([2]),
    ]
    s2 = sharded_variable.ShardedVariable(variables2)

    trace_count = [0]

    @def_function.function
    def func(sharded_var):
      trace_count[0] = trace_count[0] + 1
      sharded_var.assign([0, 0])

    func(s)
    self.assertAllEqual(ops.convert_to_tensor(s), [0, 0])
    self.assertEqual(trace_count[0], 1)
    func(s2)
    self.assertAllEqual(ops.convert_to_tensor(s2), [0, 0])
    self.assertEqual(trace_count[0], 1)

  def test_flatten(self):
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
    ]
    s = sharded_variable.ShardedVariable(variables)

    got = nest.flatten(s)
    self.assertIs(s, got[0])

    got = nest.flatten(s, expand_composites=True)
    self.assertAllEqual(variables, got)

  def test_tf_module(self):

    class Model(module.Module):

      def __init__(self):
        super().__init__()
        variables = [
            variables_lib.Variable([0]),
            variables_lib.Variable([1]),
        ]
        self.w = sharded_variable.ShardedVariable(variables)

    model = Model()

    self.assertLen(model.variables, 2)
    self.assertEqual(model.variables[0], [0])
    self.assertEqual(model.variables[1], [1])
    self.assertAllEqual(model.variables, model.trainable_variables)

    self.assertLen(model._trackable_children(), 1)
    self.assertIs(model._trackable_children().popitem()[1], model.w)

  def test_embedding_lookup(self):
    v = [
        variables_lib.Variable([[1., 2.], [3., 4.]]),
        variables_lib.Variable([[5., 6.], [7., 8.]]),
        variables_lib.Variable([[9., 10.]])
    ]
    sv = sharded_variable.ShardedVariable(v)

    @def_function.function
    def lookup():
      ids = constant_op.constant([0, 3, 4])
      return embedding_ops.embedding_lookup_v2(sv, ids)

    @def_function.function
    def sparse_lookup():
      sp_ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[0, 3, 4, 1],
          dense_shape=[3, 3])
      return embedding_ops.embedding_lookup_sparse_v2(sv, sp_ids, None)

    @def_function.function
    def safe_sparse_lookup():
      sp_ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[0, -1, 4, 1],
          dense_shape=[3, 3])
      sp_weights = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[1., 1., -1., 1.],
          dense_shape=[3, 3])
      return embedding_ops.safe_embedding_lookup_sparse_v2(
          sv, sp_ids, sp_weights)

    # TODO(chenkai): Add safe_sparse_lookup to the list. Currently
    # ShardedVariable is converted to a tensor in safe_sparse_lookup.
    for func in [lookup, sparse_lookup]:
      num_gather_ops = 0
      for op in func.get_concrete_function().graph.get_operations():
        if op.type == 'ResourceGather':
          num_gather_ops += 1
      self.assertEqual(
          num_gather_ops, len(v), 'Number of ResourceGather op does not match'
          ' expected, possibly due to ShardedVariable accidentally being'
          ' converted to tensor in embedding_lookup ops.')

    self.assertAllEqual(lookup(), [[1., 2.], [7., 8.], [9., 10.]])
    self.assertAllClose(sparse_lookup(), [[4., 5.], [9., 10.], [3., 4.]])
    self.assertAllClose(safe_sparse_lookup(), [[1., 2.], [0., 0.], [3., 4.]])

  def test_slicing(self):
    v = [
        variables_lib.Variable([[1, 2], [3, 4], [5, 6]]),
        variables_lib.Variable([[7, 8], [9, 10], [11, 12]]),
        variables_lib.Variable([[13, 14], [15, 16]])
    ]
    sv = sharded_variable.ShardedVariable(v)
    empty = v[0][0:0]

    # Test cases: positive step
    self.assertAllEqual(sv[:], array_ops.concat(v, axis=0))
    self.assertAllEqual(sv[:2], [[1, 2], [3, 4]])
    self.assertAllEqual(sv[-8:2], [[1, 2], [3, 4]])
    self.assertAllEqual(sv[-10:2], [[1, 2], [3, 4]])
    self.assertAllEqual(sv[5:], [[11, 12], [13, 14], [15, 16]])
    self.assertAllEqual(sv[5:-1], [[11, 12], [13, 14]])
    self.assertAllEqual(sv[::3], [[1, 2], [7, 8], [13, 14]])
    self.assertAllEqual(sv[::5], [[1, 2], [11, 12]])
    self.assertAllEqual(sv[1::6], [[3, 4], [15, 16]])
    self.assertAllEqual(sv[1:5:6], [[3, 4]])
    self.assertAllEqual(sv[1::7], [[3, 4]])
    self.assertAllEqual(sv[2:7], [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
    self.assertAllEqual(sv[2:7:2], [[5, 6], [9, 10], [13, 14]])
    self.assertAllEqual(sv[2:7:3], [[5, 6], [11, 12]])

    # Test cases: negative step
    self.assertAllEqual(
        sv[::-1], array_ops.reverse(array_ops.concat(v, axis=0), axis=[0]))
    self.assertAllEqual(sv[2::-1], [[5, 6], [3, 4], [1, 2]])
    self.assertAllEqual(sv[2:-8:-1], [[5, 6], [3, 4]])
    self.assertAllEqual(sv[2:-10:-1], [[5, 6], [3, 4], [1, 2]])
    self.assertAllEqual(sv[4::-1], [[9, 10], [7, 8], [5, 6], [3, 4], [1, 2]])
    self.assertAllEqual(sv[-1:-3:-1], [[15, 16], [13, 14]])
    self.assertAllEqual(sv[::-5], [[15, 16], [5, 6]])
    self.assertAllEqual(sv[6::-6], [[13, 14], [1, 2]])
    self.assertAllEqual(sv[6:5:-6], [[13, 14]])
    self.assertAllEqual(sv[6::-7], [[13, 14]])
    self.assertAllEqual(sv[7:1:-1],
                        [[15, 16], [13, 14], [11, 12], [9, 10], [7, 8], [5, 6]])
    self.assertAllEqual(sv[7:1:-2], [[15, 16], [11, 12], [7, 8]])
    self.assertAllEqual(sv[7:1:-4], [[15, 16], [7, 8]])

    # Test cases: empty slice
    self.assertAllEqual(sv[0:0], empty)
    self.assertAllEqual(sv[5:3], empty)
    self.assertAllEqual(sv[3:5:-1], empty)
    self.assertAllEqual(sv[-1:0], empty)
    self.assertAllEqual(sv[2:-1:-1], empty)

    # Test cases: slicing other dimensions
    self.assertAllEqual(sv[:, 0], [1, 3, 5, 7, 9, 11, 13, 15])
    self.assertAllEqual(sv[:, 0:1], [[1], [3], [5], [7], [9], [11], [13], [15]])

    # Test cases: normal indexing
    self.assertAllEqual(sv[2], [5, 6])
    self.assertAllEqual(sv[6], [13, 14])
    self.assertAllEqual(sv[2, 1], 6)
    self.assertAllEqual(sv[-2], [13, 14])
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      _ = sv[100]
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      _ = sv[-100]

    # Test cases: Ellipsis
    self.assertAllEqual(sv[...], array_ops.concat(v, axis=0))
    self.assertAllEqual(sv[..., 0], [1, 3, 5, 7, 9, 11, 13, 15])
    self.assertAllEqual(sv[0:1, ...], [[1, 2]])

    # Test cases: newaxis
    self.assertAllEqual(
        sv[array_ops.newaxis, ...],
        array_ops.expand_dims_v2(array_ops.concat(v, axis=0), axis=0))

    # Test cases: boolean masks
    self.assertAllEqual(sv[ops.convert_to_tensor(sv) > 10],
                        [11, 12, 13, 14, 15, 16])

    # Test cases: tensor input
    with self.assertRaisesRegex(TypeError, 'not allowed'):
      _ = sv[constant_op.constant(1)::]
    with self.assertRaisesRegex(TypeError, 'not allowed'):
      _ = sv[:constant_op.constant(1):]
    with self.assertRaisesRegex(TypeError, 'not allowed'):
      _ = sv[constant_op.constant(1)]

    # Test cases: inside tf.function
    @def_function.function
    def func():
      a = sv[:, 0]
      return a

    self.assertAllEqual(func(), [1, 3, 5, 7, 9, 11, 13, 15])

  def test_operator_overload(self):
    v1 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv1 = sharded_variable.ShardedVariable(v1)

    v2 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv2 = sharded_variable.ShardedVariable(v2)

    equal = sv1 == sv2
    self.assertAllEqual(equal, [True, True])
    self.assertAllEqual(sv1 + sv2, [2.0, 4.0])

  def test_shards_have_container_set(self):
    v1 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv1 = sharded_variable.ShardedVariable(v1)
    for v in sv1.variables:
      self.assertTrue(hasattr(v, '_sharded_container'))
      self.assertIs(v._sharded_container(), sv1)

  def test_numpy(self):
    v1 = [
        variables_lib.Variable([1.]),
        variables_lib.Variable([2.]),
    ]
    sv1 = sharded_variable.ShardedVariable(v1)
    sv1_np = sv1.numpy()
    self.assertIsInstance(sv1_np, np.ndarray)
    self.assertAllEqual(sv1_np, np.array([1., 2.]))


class ShardedVariableSaveLoadTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    cluster_def = get_cluster_def(test_cluster_params, num_workers=2, num_ps=3)
    self.cluster_resolver = SimpleClusterResolver(ClusterSpec(cluster_def))

  def tearDown(self):
    super().tearDown()
    # Reset context to disconnect from the cluster.
    context._reset_context()

  def _create_strategy(self, num_shards):
    if num_shards > 1:
      strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
          self.cluster_resolver,
          variable_partitioner=sharded_variable.FixedShardsPartitioner(
              num_shards))
    else:
      strategy = ds_context._get_default_strategy()
    return strategy

  @combinations.generate(
      combinations.combine(
          shard_config=[[2, 2], [2, 3], [3, 2], [2, 1], [1, 1]],
      ))
  def testSaveAndLoadSingleVariable(self, shard_config):
    """Test saving and loading ShardedVariable with different numbers of shards.

    Loading tf.Variables into multiple Shards is not yet supported

    Args:
      shard_config: The number of shards to use before and after loading. For
        example, [2, 1] means to create and save the variable with 2 shards and
        load it into 1 shard (i.e., a regular tf.Variable).
    """
    strategy = self._create_strategy(shard_config[0])

    with strategy.scope():
      var = variables_lib.Variable([1., 2., 3., 4., 5., 6.])

    # Save variable
    model_dir = self.get_temp_dir()
    save.save(var, model_dir)

    strategy2 = self._create_strategy(shard_config[1])
    with strategy2.scope():
      # Load variable
      loaded = load.load(model_dir)

    # Assert all values loaded, values are same
    if shard_config[1] > 1:
      loaded = array_ops.concat(loaded.variables, axis=0)
    self.assertLen(loaded.numpy(), 6)

    if shard_config[0] > 1:
      var = array_ops.concat(var.variables, axis=0)
    self.assertAllClose(var.numpy(), loaded.numpy())

  def testSaveAndLoadModuleUnderStrategy(self):

    class Dense(module.Module):

      def __init__(self):
        self.kernel = variables_lib.Variable(
            random_ops.random_uniform((6, 6)), name='kernel')
        self.bias = variables_lib.Variable(
            random_ops.random_uniform((6,)), name='bias')

      @def_function.function
      def __call__(self, x):
        out = math_ops.matmul(self.kernel, x)
        out = out + self.bias
        return out

    x = constant_op.constant(
        math_ops.range(6, dtype=dtypes.float32), shape=[6, 1])

    strategy = self._create_strategy(2)
    with strategy.scope():
      layer = Dense()
      expect = layer(x)

    model_dir = self.get_temp_dir()
    save.save(layer, model_dir)

    strategy2 = self._create_strategy(3)
    with strategy2.scope():
      loaded_layer = load.load(model_dir)
      # Should fail with informative error
      with self.assertRaisesRegex(ValueError, 'run a loaded non-Keras'):
        got = loaded_layer(x)

    # Loading without a strategy should work, because the tf.function is traced
    # with a single variable as input
    loaded_layer = load.load(model_dir)
    got = loaded_layer(x)
    self.assertAllClose(got, expect)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
