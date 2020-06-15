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
"""Tests for TPUStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_strategy_util


FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_strategy_util.initialize_tpu_system(resolver)
  return tpu_lib.TPUStrategy(resolver)


class TPUStrategyTest(test.TestCase):

  def test_multiple_initialize_system(self):
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    tpu_strategy_util.initialize_tpu_system(resolver)

    with test.mock.patch.object(logging, "warning") as mock_log:
      tpu_strategy_util.initialize_tpu_system(resolver)
      self.assertRegex(str(mock_log.call_args), "already been initialized")

  def test_sequential_experimental_runs(self):
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    topology = tpu_strategy_util.initialize_tpu_system(resolver)
    # Computation replicated to all cores.
    device_assignment = device_assignment_lib.DeviceAssignment.build(
        topology, num_replicas=2)
    strategy = tpu_lib.TPUStrategy(
        resolver, device_assignment=device_assignment)

    # Computation on the 1st core.
    device_assignment2 = device_assignment_lib.DeviceAssignment.build(
        topology, num_replicas=1)
    strategy2 = tpu_lib.TPUStrategy(
        resolver, device_assignment=device_assignment2)

    def computation(x):
      return math_ops.square(x)

    @def_function.function
    def train_step():
      outputs = strategy.experimental_local_results(
          strategy.run(computation, args=([2., 2.],)))
      outputs2 = strategy2.run(
          computation, args=([outputs[0]],))
      return outputs2

    self.assertAllEqual([[16., 16.]], train_step())

  def test_device_switch_case(self):
    strategy = get_tpu_strategy()
    with strategy.scope():
      a = variables.Variable(1)

    inference_iteration = variables.Variable(-1)

    def inference_fn(x, i):
      return a + x + i

    @def_function.function
    def run_inference(x):

      def do_inference(device, inference_fn, i):
        with ops.device(device):
          return inference_fn(x, i)

      branch_fns = {
          0: (lambda: do_inference("/device:TPU:0", inference_fn, 0)),
          1: (lambda: do_inference("/device:TPU:1", inference_fn, 1)),
      }
      branch_index = inference_iteration.assign_add(1, use_locking=True) % 2
      return control_flow_ops.switch_case(branch_index, branch_fns)

    self.assertAllEqual(2., run_inference(1))  # Use TPU core 0.
    self.assertAllEqual(3., run_inference(1))  # Use TPU core 1.

  def test_recover_from_compilation_failures(self):
    # TODO(b/148150981): Stop skipping this test once recovery works
    # for non-local TPU.
    if FLAGS.tpu:
      self.skipTest("Recovery fails for non-local TPU, see b/148150981")

    # Disable automatic outside compilation.
    config.set_soft_device_placement(False)
    strategy = get_tpu_strategy()

    @def_function.function
    def compilation_failure_run():

      def computation():
        return random_ops.random_gamma([10], [0.5, 1.5])

      return strategy.run(computation)

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "TPU compilation failed"):
      compilation_failure_run()

    @def_function.function
    def good_run():

      def computation():
        return random_ops.random_normal([10])

      return strategy.run(computation)

    good_run()

  def test_dynamic_shape_with_outside_compilation_failure(self):
    # Enable automatic outside compilation.
    config.set_soft_device_placement(True)
    strategy = get_tpu_strategy()
    dataset = dataset_ops.Dataset.from_tensors(("string", 1.0)).repeat().batch(
        2, drop_remainder=False)
    dataset = strategy.experimental_distribute_dataset(dataset)
    iterator = iter(dataset)

    @def_function.function
    def train_fn(iterator):

      def step_fn(inputs):
        _, inputs = inputs
        return math_ops.reduce_sum(inputs)

      return strategy.experimental_local_results(
          strategy.run(step_fn, args=(next(iterator),)))

    with self.assertRaisesRegex(errors.InternalError, "Compilation failure"):
      logging.info(train_fn(iterator))

  def test_computation_on_subset_cores(self):
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    topology = tpu_strategy_util.initialize_tpu_system(resolver)
    all_core_strategy = tpu_lib.TPUStrategy(resolver)

    with all_core_strategy.scope():
      v = variables.Variable(0.0,
                             aggregation=variables.VariableAggregation.MEAN)

    # Computation on the 1st core.
    device_assignment = device_assignment_lib.DeviceAssignment.build(
        topology, num_replicas=1)
    first_core_strategy = tpu_lib.TPUStrategy(
        resolver, device_assignment=device_assignment)

    # Computation on the 2nd core.
    device_assignment2 = device_assignment_lib.DeviceAssignment(
        topology, [[[0, 0, 0, 1]]])
    second_core_strategy = tpu_lib.TPUStrategy(
        resolver, device_assignment=device_assignment2)

    @def_function.function
    def train_step():

      def step_fn():
        return v + 1.0

      all_core_strategy.run(step_fn)
      r1 = first_core_strategy.run(step_fn)
      r2 = second_core_strategy.run(step_fn)
      return r1 + r2

    train_step()
    self.assertAllEqual(2., train_step())

  def test_worker_devices_on_subset_cores(self):
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    topology = tpu_strategy_util.initialize_tpu_system(resolver)

    # Strategy for the 1st core.
    device_assignment = device_assignment_lib.DeviceAssignment.build(
        topology, num_replicas=1)
    first_core_strategy = tpu_lib.TPUStrategy(
        resolver, device_assignment=device_assignment)

    # Strategy for the 2nd core.
    device_assignment2 = device_assignment_lib.DeviceAssignment(
        topology, [[[0, 0, 0, 1]]])
    second_core_strategy = tpu_lib.TPUStrategy(
        resolver, device_assignment=device_assignment2)

    self.assertLen(first_core_strategy.extended.worker_devices, 1)
    self.assertEndsWith(first_core_strategy.extended.worker_devices[0],
                        "device:TPU:0")

    self.assertLen(second_core_strategy.extended.worker_devices, 1)
    self.assertEndsWith(second_core_strategy.extended.worker_devices[0],
                        "device:TPU:1")

  def test_tpu_tf_function_same_device(self):
    with ops.device("/device:TPU:0"):
      a = variables.Variable(1)

    @function.defun_with_attributes(attributes={"_noinline": True})
    def get_a_plus_one():
      return a + 1

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def foo(x):
      with ops.device("/device:TPU:0"):
        b = x + get_a_plus_one()
      return b + 1

    result = foo(a)
    self.assertAllEqual(4, result)

  def test_tpu_return_int32(self):
    with ops.device("/device:TPU:0"):
      a = variables.Variable(0)

    @def_function.function
    def foo():
      return a + 1

    @def_function.function
    def bar():
      with ops.device("/device:TPU:1"):
        return foo()

    with ops.device("/device:CPU:0"):
      result = bar() + 1
      self.assertAllEqual(result, 2)

  def test_control_output_in_while_body_fn(self):
    strategy = get_tpu_strategy()

    with strategy.scope():
      v = variables.Variable(
          0.0, aggregation=variables.VariableAggregation.MEAN)

    @def_function.function
    def train_step():

      def step_fn():
        v.assign_add(1)

      for _ in math_ops.range(2):
        strategy.run(step_fn)

    train_step()
    self.assertEqual(2.0, v.numpy())

  def test_cluster_in_graph_and_while_body_fn(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step():

      def step_fn(prev):
        s = prev + 1
        return s

      def init_fn():
        return array_ops.zeros(shape=())

      prev = strategy.run(init_fn)
      for _ in math_ops.range(10):
        prev = strategy.run(step_fn, args=(prev,))
      return strategy.reduce(reduce_util.ReduceOp.SUM, prev, axis=None)

    sum_val = train_step().numpy().astype(float)
    self.assertEqual(sum_val, strategy.num_replicas_in_sync * 10)

  def test_two_clusters_with_same_fn(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def foo(x):
      return strategy.run(lambda x: x + 1, (x,))

    @def_function.function
    def bar(x):
      foo(x)
      return foo(x)

    bar(1)

  def test_using_external_variable_inside_tf_function(self):
    strategy = get_tpu_strategy()
    dataset = dataset_ops.Dataset.range(
        strategy.num_replicas_in_sync * 2,
        output_type=dtypes.float32).batch(strategy.num_replicas_in_sync)
    input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    v = variables.Variable(2.0)

    @def_function.function
    def train_step(data):
      def computation(inputs):
        return inputs + v
      return strategy.run(computation, args=(data,))

    expected_result = [[x + 2.] for x in range(0, strategy.num_replicas_in_sync)
                      ]
    self.assertAllEqual(
        expected_result,
        strategy.experimental_local_results(train_step(next(input_iterator))))

  def test_keras_metric_outside_strategy_scope_per_replica(self):
    strategy = get_tpu_strategy()
    metric = keras.metrics.Mean("test_metric", dtype=dtypes.float32)

    dataset = dataset_ops.Dataset.range(strategy.num_replicas_in_sync *
                                        2).batch(2)
    dataset = strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def step_fn(i):
      metric.update_state(i)

    with self.assertRaisesRegex(ValueError, "Trying to run metric.update_state "
                                            "in replica context"):
      with strategy.scope():
        for i in dataset:
          strategy.run(step_fn, args=(i,))

  # TODO(b/145574622): Remove this test once it is re-enabled in values_test.py.
  def test_all_reduce_on_sync_on_read_variable(self):
    strategy = get_tpu_strategy()
    dataset = dataset_ops.Dataset.range(
        strategy.num_replicas_in_sync, output_type=dtypes.float32).batch(
            strategy.num_replicas_in_sync, drop_remainder=True)
    input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    with strategy.scope():
      w = variables.Variable(
          (0.,),
          shape=(1,),
          trainable=False,
          synchronization=variables.VariableSynchronization.ON_READ,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

    @def_function.function
    def run(iterator):

      def computation(x):
        w.assign(x + w)
        return w

      def all_reduce(x):
        ctx = distribution_strategy_context.get_replica_context()
        return ctx.all_reduce("SUM", w) + x

      outputs = strategy.run(computation, args=(next(iterator),))
      outputs2 = strategy.experimental_local_results(
          strategy.run(all_reduce, args=(outputs,)))
      return outputs2

    data = range(0, strategy.num_replicas_in_sync)
    data_sum = sum(data)
    expected_result = [
        [x + data_sum] for x in range(0, strategy.num_replicas_in_sync)
    ]
    self.assertAllEqual(expected_result, run(input_iterator))
    self.assertAllEqual((0.,), w.read_value())

  # TODO(b/140633529): Re-enable the test.
  def disable_test_experimental_run_output_on_device(self):
    strategy = get_tpu_strategy()

    def computation(x):
      return math_ops.square(x)

    @def_function.function
    def train_step():
      outputs = strategy.experimental_local_results(
          strategy.run(computation, args=(2,)))
      return outputs

    results = train_step()
    self.assertAllEqual([4., 4.], results)
    self.assertAllEqual("/job:localhost/replica:0/task:0/device:TPU:0",
                        results[0].backing_device)
    self.assertAllEqual("/job:localhost/replica:0/task:0/device:TPU:1",
                        results[1].backing_device)

  def test_composite_input(self):
    strategy = get_tpu_strategy()
    if strategy.num_replicas_in_sync != 2:
      self.skipTest("Test assumes two replicas.")

    with strategy.scope():
      table = variables.Variable(
          initial_value=[[0.0, 1.0], [3.0, 7.0]], dtype=dtypes.float32)

    @def_function.function
    def sparse_lookup(iterator):

      def tpu_function(sparse):
        # Assumes dense_shape is (2, *)
        looked_up = array_ops.gather(table, sparse.values)
        return math_ops.unsorted_segment_sum(looked_up, sparse.indices[:, 0], 2)

      return strategy.experimental_local_results(
          strategy.run(tpu_function, args=(next(iterator),)))

    def dataset_fn(_):
      dataset = dataset_ops.Dataset.range(2)

      def make_sparse(_):
        return sparse_tensor.SparseTensor(
            indices=array_ops.constant([[0, 0], [1, 0], [1, 1]],
                                       dtype=dtypes.int64),
            values=array_ops.constant([0, 0, 1], dtype=dtypes.int32),
            dense_shape=array_ops.constant([2, 2], dtype=dtypes.int64))

      return dataset.map(make_sparse)

    strategy.extended._set_prefetch_on_host(True)  # pylint: disable=protected-access
    dataset = iter(
        strategy.experimental_distribute_datasets_from_function(dataset_fn))

    result = sparse_lookup(dataset)
    self.assertAllEqual(result,
                        [[[0.0, 1.0], [3.0, 8.0]], [[0.0, 1.0], [3.0, 8.0]]])

  def test_composite_input_dynamic_shapes_outside_compilation(self):
    strategy = get_tpu_strategy()
    if strategy.num_replicas_in_sync != 2:
      self.skipTest("Test assumes two replicas.")

    table = variables.Variable(
        initial_value=[[0.0, 1.0], [3.0, 7.0]], dtype=dtypes.float32)

    @def_function.function
    def sparse_lookup(iterator):

      def tpu_function(sparse):
        lookup = tpu.outside_compilation(
            embedding_ops.safe_embedding_lookup_sparse, table, sparse)
        return math_ops.reduce_sum(lookup, axis=0)

      return strategy.experimental_local_results(
          strategy.run(tpu_function, args=(next(iterator),)))

    def dataset_fn(_):
      dataset = dataset_ops.Dataset.range(2)

      def make_sparse(i):
        indices = array_ops.constant([[0, 0], [1, 0], [1, 1]],
                                     dtype=dtypes.int64)[0:2 + i]
        values = array_ops.constant([0, 0, 1], dtype=dtypes.int32)[0:2 + i]
        shape = [
            array_ops.constant([2], dtype=dtypes.int64),
            array_ops.expand_dims(1 + i, axis=0)
        ]
        dense_shape = array_ops.concat(shape, axis=0)
        return sparse_tensor.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape)

      return dataset.map(make_sparse)

    strategy.extended._set_prefetch_on_host(True)  # pylint: disable=protected-access
    dataset = iter(
        strategy.experimental_distribute_datasets_from_function(dataset_fn))

    result = sparse_lookup(dataset)
    self.assertAllEqual(result, [[0.0, 2.0], [1.5, 5.0]])

  def test_per_device_tracing_of_mirrored_variables(self):
    # Define trace_count as a list to avoid python scoping error
    trace_count = [0]

    strategy = get_tpu_strategy()
    with strategy.scope():
      variable = variables.Variable(0.0)

    @def_function.function
    def add_one():
      trace_count[0] = trace_count[0] + 1
      return math_ops.add(variable, constant_op.constant(1.0))

    @def_function.function
    def update_variable():
      for device in set(strategy.extended.worker_devices):
        with ops.device(device):
          add_one()

    with strategy.scope():
      update_variable.get_concrete_function()
      self.assertEqual(trace_count[0], len(strategy.extended.worker_devices))


if __name__ == "__main__":
  test.main()
