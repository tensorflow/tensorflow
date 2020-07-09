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
"""Test DistributionStrategy, ReplicaContext, and supporting APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest


class _TestReplicaContext(distribute_lib.ReplicaContext):

  def merge_call(self, fn, *args, **kwargs):
    return kwargs["test_arg"]


def _get_test_variable(name, synchronization, aggregation):
  return {
      "name": name,
      "synchronization": synchronization,
      "aggregation": aggregation
  }


def _test_input_fn(input_context):
  del input_context
  return dataset_ops.DatasetV2.from_tensors(1.).repeat()


class _TestStrategy(distribute_lib.Strategy):

  def __init__(self):
    super(_TestStrategy, self).__init__(_TestExtended(self))


class _TestExtended(distribute_lib.StrategyExtendedV1):

  def __init__(self, distribute):
    super(_TestExtended, self).__init__(distribute)
    worker_device_pairs = [("", ["/device:CPU:0"])]
    self._input_workers = input_lib.InputWorkers(worker_device_pairs)

  def _call_for_each_replica(self, fn, args, kwargs):
    with _TestReplicaContext(
        self._container_strategy(),
        replica_id_in_sync_group=constant_op.constant(0, dtypes.int32)):
      return fn(*args, **kwargs)

  def _create_variable(self, next_creator, **kwargs):
    return _get_test_variable(kwargs["name"], kwargs["synchronization"],
                              kwargs["aggregation"])

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [distribute_lib.InputContext()],
                                           self._container_strategy())

  def _experimental_distribute_datasets_from_function(self, dataset_fn,
                                                      options):
    return dataset_fn(distribute_lib.InputContext())

  def _local_results(self, value):
    return (value,)

  def _reduce_to(self, reduce_op, value, destinations, experimental_hints):
    del reduce_op, destinations, experimental_hints
    return value

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    del session
    return dataset_ops.DatasetV2.from_tensor_slices(numpy_input)

  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    # TODO(tomhennigan) This is missing many things (e.g. ctx.run_op).
    ctx = input_lib.MultiStepContext()
    for _ in range(iterations):
      fn(ctx, iterator.get_next())
    return ctx

  def _update(self, var, fn, args, kwargs, group):
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    del colocate_with
    result = fn(*args, **kwargs)
    if group:
      return result
    else:
      return nest.map_structure(self._unwrap, result)


def _assert_in_default_state(t):
  t.assertIs(ds_context._get_default_replica_context(),
             ds_context.get_replica_context())
  t.assertIs(None, ds_context.get_cross_replica_context())
  t.assertFalse(ds_context.in_cross_replica_context())
  t.assertIs(ds_context._get_default_strategy(), ds_context.get_strategy())
  t.assertFalse(ds_context.has_strategy())


def _run_in_and_out_of_scope(unbound_test_method):
  def wrapper(test_case):
    dist = _TestStrategy()
    # Running in the default (replica) scope should be supported.
    _assert_in_default_state(test_case)
    unbound_test_method(test_case, dist)
    # As well as running in the strategy scope.
    with dist.scope():
      unbound_test_method(test_case, dist)
    _assert_in_default_state(test_case)
    # When run under a different strategy the test method should fail.
    another_strategy = _TestStrategy()
    msg = "Mixing different .*Strategy objects"
    with test_case.assertRaisesRegex(RuntimeError, msg):
      with another_strategy.scope():
        unbound_test_method(test_case, dist)
  return wrapper


class TestStrategyTest(test.TestCase):

  def testCallForEachReplica(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()

    def run_fn():
      replica_context = ds_context.get_replica_context()
      self.assertTrue(replica_context is not None)
      self.assertIs(None, ds_context.get_cross_replica_context())
      self.assertFalse(ds_context.in_cross_replica_context())
      self.assertTrue(ds_context.has_strategy())
      self.assertIs(dist, ds_context.get_strategy())
      self.assertEqual("foo", replica_context.merge_call(None, test_arg="foo"))
      expected_value = _get_test_variable(
          "bar", variable_scope.VariableSynchronization.AUTO,
          variable_scope.VariableAggregation.NONE)
      self.assertDictEqual(expected_value,
                           variable_scope.variable(1.0, name="bar"))

    dist.extended.call_for_each_replica(run_fn)
    with dist.scope():
      dist.extended.call_for_each_replica(run_fn)
    _assert_in_default_state(self)

  def testScope(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      self.assertIs(None, ds_context.get_replica_context())
      self.assertIs(dist, ds_context.get_cross_replica_context())
      self.assertTrue(ds_context.in_cross_replica_context())
      self.assertTrue(ds_context.has_strategy())
      self.assertIs(dist, ds_context.get_strategy())
      expected_value = _get_test_variable(
          "baz", variable_scope.VariableSynchronization.AUTO,
          variable_scope.VariableAggregation.NONE)
      self.assertDictEqual(expected_value,
                           variable_scope.variable(1.0, name="baz"))
    _assert_in_default_state(self)

  def testScopeDeviceNestingError(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    # Open a device scope with dist.scope().
    dist.extended._default_device = "/device:GPU:0"
    scope = dist.scope()
    scope.__enter__()
    self.assertIs(dist, ds_context.get_strategy())
    with ops.device("/device:CPU:0"):
      with self.assertRaisesRegex(RuntimeError, "Device scope nesting error"):
        scope.__exit__(None, None, None)
    scope.__exit__(None, None, None)
    _assert_in_default_state(self)

  def testScopeVarCreatorNestingError(self):

    def creator(next_creator, **kwargs):
      return next_creator(**kwargs)

    _assert_in_default_state(self)
    dist = _TestStrategy()
    scope = dist.scope()
    scope.__enter__()
    self.assertIs(dist, ds_context.get_strategy())
    with variable_scope.variable_creator_scope(creator):
      with self.assertRaisesRegex(RuntimeError,
                                  "Variable creator scope nesting error"):
        scope.__exit__(None, None, None)
    scope.__exit__(None, None, None)
    _assert_in_default_state(self)

  def testScopeVarScopeNestingError(self):
    # We create a new graph here to simplify clean-up, since the error
    # we are triggering happens in the middle of scope.__exit__() and
    # leaves us in a weird state.
    with ops.Graph().as_default():
      _assert_in_default_state(self)
      dist = _TestStrategy()
      scope = dist.scope()
      scope.__enter__()
      self.assertIs(dist, ds_context.get_strategy())
      with variable_scope.variable_scope("AA"):
        with self.assertRaisesRegex(RuntimeError,
                                    "Variable scope nesting error"):
          scope.__exit__(None, None, None)
    _assert_in_default_state(self)

  def testSettingSynchronizationAndAggregation(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      expected_value = _get_test_variable(
          "baz", variable_scope.VariableSynchronization.ON_WRITE,
          variable_scope.VariableAggregation.MEAN)
      self.assertDictEqual(
          expected_value,
          variable_scope.variable(
              1.0,
              name="baz",
              synchronization=variable_scope.VariableSynchronization.ON_WRITE,
              aggregation=variable_scope.VariableAggregation.MEAN))
    _assert_in_default_state(self)

  def testSetStrategy(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    dist2 = _TestStrategy()
    ds_context.experimental_set_strategy(dist)
    self.assertIs(None, ds_context.get_replica_context())
    self.assertIs(dist, ds_context.get_cross_replica_context())
    self.assertTrue(ds_context.in_cross_replica_context())
    self.assertTrue(ds_context.has_strategy())
    self.assertIs(dist, ds_context.get_strategy())
    expected_value = _get_test_variable(
        "baz", variable_scope.VariableSynchronization.AUTO,
        variable_scope.VariableAggregation.NONE)
    self.assertDictEqual(expected_value,
                         variable_scope.variable(1.0, name="baz"))
    ds_context.experimental_set_strategy(dist2)
    self.assertIs(dist2, ds_context.get_strategy())
    ds_context.experimental_set_strategy(None)
    _assert_in_default_state(self)

  def testSetStrategyInScope(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      with self.assertRaisesRegex(
          RuntimeError,
          "Must not be called inside a `tf.distribute.Strategy` scope"):
        ds_context.experimental_set_strategy(_TestStrategy())
      with self.assertRaisesRegex(
          RuntimeError,
          "Must not be called inside a `tf.distribute.Strategy` scope"):
        ds_context.experimental_set_strategy(dist)
      with self.assertRaisesRegex(
          RuntimeError,
          "Must not be called inside a `tf.distribute.Strategy` scope"):
        ds_context.experimental_set_strategy(None)
    _assert_in_default_state(self)

  def testSameScopeNesting(self):
    _assert_in_default_state(self)
    dist = _TestStrategy()
    scope_a = dist.scope()
    with scope_a:
      self.assertIs(dist, ds_context.get_strategy())
      scope_b = dist.scope()
      with scope_b:
        self.assertIs(dist, ds_context.get_strategy())
        with scope_a:
          self.assertIs(dist, ds_context.get_strategy())
        self.assertIs(dist, ds_context.get_strategy())
      self.assertIs(dist, ds_context.get_strategy())
      dist2 = _TestStrategy()
      scope2 = dist2.scope()
      with self.assertRaisesRegex(
          RuntimeError, "Mixing different tf.distribute.Strategy objects"):
        with scope2:
          pass
    _assert_in_default_state(self)
    with scope_b:
      self.assertIs(dist, ds_context.get_strategy())
    _assert_in_default_state(self)

  @_run_in_and_out_of_scope
  def testMakeInputFnIterator(self, dist):
    self.assertIsNotNone(dist.make_input_fn_iterator(_test_input_fn))

  @_run_in_and_out_of_scope
  def testReduce(self, dist):
    x = constant_op.constant(1.)
    x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
    self.assertEqual(self.evaluate(x), self.evaluate(x_r))

  def testReductions_acceptStringOps(self):
    dist = _TestStrategy()
    for op in ("mean", "MEAN", "sum", "SUM"):
      x = constant_op.constant(1.)
      y = constant_op.constant(1.)
      x_r = dist.reduce(op, x, axis=None)
      self.assertEqual(self.evaluate(x), self.evaluate(x_r))
      x_r = dist.extended.reduce_to(op, x, "/CPU:0")
      self.assertEqual(self.evaluate(x), self.evaluate(x_r))
      x_r, y_r = dist.extended.batch_reduce_to(op,
                                               ((x, "/CPU:0"), (y, "/CPU:0")))
      self.assertEqual(self.evaluate(x), self.evaluate(x_r))
      self.assertEqual(self.evaluate(y), self.evaluate(y_r))

  @_run_in_and_out_of_scope
  def testExperimentalMakeNumpyDataset(self, dist):
    numpy_input = np.ones([10], dtype=np.float32)
    dataset = dist.experimental_make_numpy_dataset(numpy_input)
    self.assertEqual(
        self.evaluate(dataset.reduce(0., lambda a, b: a + b)), 10.)

  @_run_in_and_out_of_scope
  def testExperimentalRunStepsOnIterator(self, dist):
    all_inputs = []
    dataset = dataset_ops.Dataset.from_tensors(1.).repeat()
    dist.extended.experimental_run_steps_on_iterator(
        lambda _, inputs: all_inputs.append(self.evaluate(inputs)),
        dataset_ops.make_one_shot_iterator(dataset))
    self.assertEqual(all_inputs, [1.])

  @_run_in_and_out_of_scope
  def testReduceTo(self, dist):
    x = constant_op.constant(1.)
    x_r = dist.extended.reduce_to(reduce_util.ReduceOp.MEAN, x, "/CPU:0")
    self.assertEqual(self.evaluate(x), self.evaluate(x_r))

  @_run_in_and_out_of_scope
  def testBatchReduceTo(self, dist):
    x = constant_op.constant(1.)
    y = constant_op.constant(1.)
    x_r, y_r = dist.extended.batch_reduce_to(reduce_util.ReduceOp.MEAN,
                                             ((x, "/CPU:0"), (y, "/CPU:0")))
    self.assertEqual(self.evaluate(x), self.evaluate(x_r))
    self.assertEqual(self.evaluate(y), self.evaluate(y_r))

  @_run_in_and_out_of_scope
  def testUpdate(self, dist):
    with dist.scope():
      v = variables.Variable(1.)
    t = constant_op.constant(2.)

    def assign_fn(vv, tt):
      self.assertIs(vv, v)
      self.assertIs(tt, t)
    dist.extended.update(v, assign_fn, (t,))

  @_run_in_and_out_of_scope
  def testUpdateAutoGraph(self, dist):
    with dist.scope():
      v = variables.Variable(1.)
    t = constant_op.constant(2.)

    def assign_fn(unused_vv, unused_tt):
      self.assertTrue(converter_testing.is_inside_generated_code())

    @def_function.function  # AutoGraph is default-on only within tf.function
    def test_fn():
      dist.extended.update(v, assign_fn, (t,))

    test_fn()

  @_run_in_and_out_of_scope
  def testUpdateNonSlot(self, dist):
    t = constant_op.constant(2.)
    update_calls = []
    dist.extended.update_non_slot(t, lambda: update_calls.append(1))
    self.assertEqual(len(update_calls), 1)

  @_run_in_and_out_of_scope
  def testUpdateNonSlotAutoGraph(self, dist):
    t = constant_op.constant(2.)

    def update_fn():
      self.assertTrue(converter_testing.is_inside_generated_code())

    @def_function.function  # AutoGraph is default-on only within tf.function
    def test_fn():
      dist.extended.update_non_slot(t, update_fn)

    test_fn()

  def testClusterResolverDefaultNotImplemented(self):
    dist = _TestStrategy()
    self.assertIsNone(dist.cluster_resolver)
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    cluster_resolver = SimpleClusterResolver(base_cluster_spec)
    dist.extended._cluster_resolver = cluster_resolver
    self.assertIs(dist.cluster_resolver, cluster_resolver)


# _TestStrategy2 is like _TestStrategy, except it doesn't change variable
# creation.
class _TestStrategy2(distribute_lib.Strategy):

  def __init__(self):
    super(_TestStrategy2, self).__init__(_TestExtended2(self))


class _TestExtended2(_TestExtended):

  def _create_variable(self, next_creator, **kwargs):
    return next_creator(**kwargs)


class DefaultDistributionStrategyTest(test.TestCase, parameterized.TestCase):

  def testMergeCall(self):
    _assert_in_default_state(self)

    def merge_fn(dist, s):
      self.assertIs(ds_context._get_default_strategy(), dist)
      self.assertIs(None, ds_context.get_replica_context())
      self.assertIs(dist, ds_context.get_cross_replica_context())
      self.assertTrue(ds_context.in_cross_replica_context())
      self.assertIs(dist, ds_context.get_strategy())
      self.assertFalse(ds_context.has_strategy())
      return "foo_" + s

    replica_ctx = ds_context.get_replica_context()
    self.assertIs(ds_context._get_default_replica_context(), replica_ctx)
    self.assertEqual("foo_bar", replica_ctx.merge_call(merge_fn, args=("bar",)))
    _assert_in_default_state(self)

  def testMergeCallAutoGraph(self):
    _assert_in_default_state(self)

    def merge_fn(_, s):
      self.assertTrue(converter_testing.is_inside_generated_code())
      return s

    @def_function.function  # AutoGraph is default-on only within tf.function
    def test_fn():
      replica_ctx = ds_context.get_replica_context()
      replica_ctx.merge_call(merge_fn, args=("bar",))

    test_fn()

  def testScopeMostlyNoOp(self):
    _assert_in_default_state(self)

    test_strategy = _TestStrategy2()
    with test_strategy.scope():
      variable_scope.variable(1.0, name="before")

    default_strategy = ds_context._get_default_strategy()
    scope = default_strategy.scope()
    with scope:
      _assert_in_default_state(self)

      with test_strategy.scope():
        with self.assertRaisesRegex(
            RuntimeError, "Mixing different tf.distribute.Strategy objects"):
          variable_scope.variable(1.0, name="error")

      with scope:
        _assert_in_default_state(self)

        with test_strategy.scope():
          with self.assertRaisesRegex(
              RuntimeError, "Mixing different tf.distribute.Strategy objects"):
            variable_scope.variable(1.0, name="also_error")

      _assert_in_default_state(self)

    _assert_in_default_state(self)
    with test_strategy.scope():
      variable_scope.variable(1.0, name="after")

  def testExperimentalRunV2(self):
    default_strategy = ds_context._get_default_strategy()
    dataset = dataset_ops.Dataset.range(10).batch(2)
    iterator = default_strategy.extended._make_dataset_iterator(dataset)
    next_val = iterator.get_next()

    def train_step(input_data):
      return input_data

    for _ in range(2):
      default_strategy.run(train_step, args=(next_val,))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testDistributedDatasets(self):
    default_strategy = ds_context._get_default_strategy()
    if context.executing_eagerly():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
      dist_dataset = default_strategy.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
      next_val = next(iter(dist_dataset))
    else:
      dataset_fn = lambda _: dataset_ops.DatasetV1.range(10).batch(2)
      dist_dataset = default_strategy.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
      iterator = dist_dataset.make_initializable_iterator()
      self.evaluate(iterator.initializer)
      next_val = iterator.get_next()
    self.assertAllEqual([0, 1], self.evaluate(next_val))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testDistributedDatasetsFromFunction(self):
    default_strategy = ds_context._get_default_strategy()
    if context.executing_eagerly():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
      dist_dataset_from_func = \
          default_strategy.experimental_distribute_datasets_from_function(
              dataset_fn)
      next_val = next(iter(dist_dataset_from_func))
      self.assertAllEqual([0, 1], self.evaluate(next_val))
    else:
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
      dist_dataset_from_func = \
        default_strategy.experimental_distribute_datasets_from_function(
            dataset_fn)
      dataset_ops.make_initializable_iterator(dist_dataset_from_func)


class InputContextTest(test.TestCase):

  def testProperties(self):
    input_context = distribute_lib.InputContext(
        num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
    self.assertEqual(6, input_context.num_replicas_in_sync)
    self.assertEqual(1, input_context.input_pipeline_id)
    self.assertEqual(2, input_context.num_input_pipelines)

  def testPerReplicaBatchSize(self):
    input_context = distribute_lib.InputContext(
        num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
    self.assertEqual(2, input_context.get_per_replica_batch_size(12))
    with self.assertRaises(ValueError):
      input_context.get_per_replica_batch_size(13)

  def testStr(self):
    input_context = distribute_lib.InputContext(
        num_input_pipelines=1, input_pipeline_id=0, num_replicas_in_sync=42)
    self.assertEqual(
        "tf.distribute.InputContext(input pipeline id 0, total: 1)",
        str(input_context))
    input_context = distribute_lib.InputContext(
        num_input_pipelines=3, input_pipeline_id=1, num_replicas_in_sync=42)
    self.assertEqual(
        "tf.distribute.InputContext(input pipeline id 1, total: 3)",
        str(input_context))


if __name__ == "__main__":
  test.main()
