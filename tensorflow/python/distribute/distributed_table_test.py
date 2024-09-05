# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for distributed_table."""

import copy
import os

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import save as tf_save


source_combination = combinations.combine(source=["textfile", "keyvaluetensor"])


class DistributedTableTest(test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(DistributedTableTest, cls).setUpClass()
    cls.cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=2, num_ps=3, rpc_layer="grpc")
    cls.cluster_resolver = cls.cluster.cluster_resolver

  @classmethod
  def tearDownClass(cls):
    super(DistributedTableTest, cls).tearDownClass()
    cls.cluster.stop()

  def make_initializer(self, init_source, vals):
    if init_source == "textfile":
      file = os.path.join(self.get_temp_dir(), "text_file_initializer")
      with open(file, "w") as f:
        f.write("\n".join(str(v) for v in vals) + "\n")
      return lookup_ops.TextFileInitializer(
          filename=file,
          key_dtype=dtypes.int64,
          key_index=lookup_ops.TextFileIndex.LINE_NUMBER,
          value_dtype=dtypes.int64,
          value_index=lookup_ops.TextFileIndex.WHOLE_LINE)
    elif init_source == "keyvaluetensor":
      keys_tensor = constant_op.constant(
          list(range(len(vals))), dtype=dtypes.int64)
      vals_tensor = constant_op.constant(vals, dtype=dtypes.int64)
      return lookup_ops.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    else:
      raise ValueError("Unrecognized init_source: " + init_source)

  def createStaticHashTable(self,
                            init_source=None,
                            vals=None,
                            default_value=None,
                            initializer=None):
    if not initializer:
      initializer = self.make_initializer(init_source, vals)
    return lookup_ops.StaticHashTable(
        initializer=initializer, default_value=default_value)

  def makeDatasetFromTensorWithoutUsingResource(self, input_context, tensor):
    """Returns a dataset made from `tensor`. To be called in a dataset_fn."""
    global_batch_size = 24
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset_ops.DatasetV2.from_tensors(tensor).repeat().batch(
        batch_size, drop_remainder=True)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
    dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
    return dataset

  @combinations.generate(source_combination)
  def testCreateDistributedTableInScope(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    self.assertIsInstance(lookuptable, ps_values.DistributedTable)
    self.assertEqual(self.evaluate(lookuptable.size()), 3)

    # Lookup on the coordinator.
    output = lookuptable.lookup(
        constant_op.constant([0, 1, -1], dtype=dtypes.int64))
    self.assertAllEqual([0, 1, -2], output)
    self.assertEqual(lookuptable.size(), 3)

  @combinations.generate(source_combination)
  def testCopyDistributedTable(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    new_table = copy.copy(lookuptable)
    # No new coordinator instance or distributed tables are created.
    self.assertDictEqual(lookuptable.__dict__, new_table.__dict__)

  @combinations.generate(source_combination)
  def testCreateLookupInDatasetFnUnderScope(self, source):
    # TODO(wxinyi): Warn the user of the inefficiency of this workflow (i.e.
    # creating `StaticHashTable` inside a `@tf.function`-wrapped `dataset_fn` to
    # be distributed with `distribute_datasets_from_function` and
    # `create_per_worker_dataset`.
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():

      def dataset_fn(input_context):
        some_out_of_range_tensor = constant_op.constant(10, dtype=dtypes.int64)
        lookuptable = self.createStaticHashTable(
            init_source=source, vals=[0, 1, 2], default_value=-2)

        self.assertNotIsInstance(lookuptable, ps_values.DistributedTable)

        generation_tensor = lookuptable.lookup(some_out_of_range_tensor)
        dataset = self.makeDatasetFromTensorWithoutUsingResource(
            input_context, generation_tensor)
        return dataset

      @def_function.function
      def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(dataset_fn)

      per_worker_dataset = coordinator.create_per_worker_dataset(
          per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)

      @def_function.function
      def worker_fn(iterator):
        return math_ops.reduce_sum(next(iterator))

      result = []
      for _ in range(10):
        result.append(
            coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

      for r in result:
        returned_input = r.fetch()
        self.assertAllClose(-48, returned_input)

  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithoutMap(self, source):

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    def dataset_fn(input_context):
      some_out_of_range_tensor = constant_op.constant(10, dtype=dtypes.int64)

      self.assertIsInstance(lookuptable, ps_values.DistributedTable)

      generation_tensor = lookuptable.lookup(some_out_of_range_tensor)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      return dataset

    @def_function.function
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(dataset_fn)

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-48, returned_input)

  @combinations.generate(
      combinations.combine(
          source=["textfile", "keyvaluetensor"],
          create_datasets_under_scope=[True, False],
          using_dataset_instance_not_function=[True, False],
          create_per_worker_dataset_takes_instance=[True, False]))
  def testCreateTableUnderScopeCombo(self, source,
                                     create_datasets_under_scope,
                                     using_dataset_instance_not_function,
                                     create_per_worker_dataset_takes_instance):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookup_table = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    if using_dataset_instance_not_function:

      def per_worker_dataset_fn():
        dataset = dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64))
        dataset = dataset.repeat().batch(24, drop_remainder=True).prefetch(2)
        dataset = dataset.map(lookup_table.lookup)

        return strategy.experimental_distribute_dataset(dataset)

    else:

      def per_worker_dataset_fn():
        def dataset_fn(input_context):
          batch_size = input_context.get_per_replica_batch_size(24)
          dataset = dataset_ops.DatasetV2.from_tensors(
              constant_op.constant([0, 1, 3], dtype=dtypes.int64))
          dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
          dataset = dataset.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
          dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
          dataset = dataset.map(lookup_table.lookup)
          return dataset
        return strategy.distribute_datasets_from_function(dataset_fn)

    if create_datasets_under_scope:
      with strategy.scope():
        if create_per_worker_dataset_takes_instance:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn())
        else:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn)
        per_worker_iterator = iter(per_worker_dataset)

    else:
      if create_per_worker_dataset_takes_instance:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn())
      else:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  @combinations.generate(
      combinations.combine(
          source=["textfile", "keyvaluetensor"],
          create_datasets_under_scope=[True, False],
          using_dataset_instance_not_function=[True, False],
          create_per_worker_dataset_takes_instance=[True, False]))
  def testCreateTableInDatasetCombo(self, source, create_datasets_under_scope,
                                    using_dataset_instance_not_function,
                                    create_per_worker_dataset_takes_instance):

    if using_dataset_instance_not_function and (
        not create_per_worker_dataset_takes_instance):
      # This is the case that uses the `experimental_distribute_dataset` API to
      # distribute dataset (instead of the `distribute_datasets_from_function`
      # API), and passes `create_per_worker_dataset` a function that returns
      # the distributed dataset (instead of passing it the distributed dataset
      # directly).
      # TODO(b/201775366): evaluate whether we need to handle this case
      self.skipTest("Failed to serialize the input pipeline graph")

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    if using_dataset_instance_not_function:

      def per_worker_dataset_fn():
        # If this line is being called under strategy.scope(), it becomes a
        # DistributedTable. Interestingly, after
        # `experimental_distribute_dataset` serializes the dataset on chief and
        # deserializes it on workers, `lookup_table` becomes a
        # RestoredDistributedTable instead of a DistributedTable. And when it’s
        # `resource_handle` is being accessed on the worker, it does not detect
        # a DispatchContext, so it returns the restored resource handle,
        # which is also the one on the local worker. The LookupTableFindV2 ops
        # is on the local worker, too.
        lookup_table = self.createStaticHashTable(
            init_source=source, vals=[0, 1, 2], default_value=-2)

        if create_datasets_under_scope:
          self.assertIsInstance(lookup_table, ps_values.DistributedTable)

        dataset = dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64))
        dataset = dataset.repeat().batch(24, drop_remainder=True).prefetch(2)
        dataset = dataset.map(lookup_table.lookup)

        return strategy.experimental_distribute_dataset(dataset)

    else:

      def per_worker_dataset_fn():

        def dataset_fn(input_context):
          # When we're wrapping the initialization of a StaticHashTable inside a
          # `dataset_fn` to be distributed with
          # `distribute_datasets_from_function`, no matter it's called under
          # strategy.scope() or not, this call creates a StaticHashTable on
          # chief instead of a DistributedTable on chief and workers.
          # And correspondingly, LookupTableFindV2 ops is on chief and there are
          # send-recv communication for the lookup.
          lookup_table = self.createStaticHashTable(
              init_source=source, vals=[0, 1, 2], default_value=-2)
          if create_datasets_under_scope:
            self.assertIsInstance(lookup_table, lookup_ops.StaticHashTable)
            self.assertNotIsInstance(lookup_table, ps_values.DistributedTable)

          batch_size = input_context.get_per_replica_batch_size(24)
          dataset = dataset_ops.DatasetV2.from_tensors(
              constant_op.constant([0, 1, 3], dtype=dtypes.int64))
          dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
          dataset = dataset.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
          dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
          dataset = dataset.map(lookup_table.lookup)
          return dataset

        return strategy.distribute_datasets_from_function(dataset_fn)

    if create_datasets_under_scope:
      with strategy.scope():
        if create_per_worker_dataset_takes_instance:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn())
        else:
          per_worker_dataset = coordinator.create_per_worker_dataset(
              per_worker_dataset_fn)
        per_worker_iterator = iter(per_worker_dataset)

    else:
      if create_per_worker_dataset_takes_instance:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn())
      else:
        per_worker_dataset = coordinator.create_per_worker_dataset(
            per_worker_dataset_fn)
      per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  @combinations.generate(source_combination)
  def testAccessingTableInStepFunction(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookup_table = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    dataset = (
        dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)).repeat().batch(
                24, drop_remainder=True).prefetch(2))
    dataset = dataset.map(lookup_table.lookup)

    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    distributed_dataset = coordinator.create_per_worker_dataset(
        distributed_dataset)

    @def_function.function
    def worker_fn(iterator):

      def replica_fn(inputs):
        return math_ops.reduce_sum(lookup_table.lookup(inputs))

      all_results = strategy.run(replica_fn, args=(next(iterator),))
      return all_results

    steps_per_epoch = 10
    distributed_iterator = iter(distributed_dataset)
    result = []
    for _ in range(steps_per_epoch):

      result.append(
          coordinator.schedule(worker_fn, args=(distributed_iterator,)))

    coordinator.join()

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  @combinations.generate(source_combination)
  def testAccessingResourceHandleInDatasetFnWithMapFnDefinedOutside(
      self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)

    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    with strategy.scope():
      lookuptable = self.createStaticHashTable(
          init_source=source, vals=[0, 1, 2], default_value=-2)

    def map_fn(vals):
      return lookuptable.lookup(vals)

    def dataset_fn(input_context):
      generation_tensor = constant_op.constant([0, 1, 3], dtype=dtypes.int64)
      dataset = self.makeDatasetFromTensorWithoutUsingResource(
          input_context, generation_tensor)
      dataset = dataset.map(map_fn)
      return dataset

    @def_function.function
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(dataset_fn)

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      return math_ops.reduce_sum(next(iterator))

    result = []
    for _ in range(10):
      # batch_size == 24 and each input is [0, 1, -2]
      result.append(
          coordinator.schedule(worker_fn, args=(per_worker_iterator,)))

    for r in result:
      returned_input = r.fetch()
      self.assertAllClose(-24, returned_input)

  class Model(module.Module):

    def __init__(self, init_source, filepath):
      vals = [0, 1, 2]
      if init_source == "textfile":

        with open(filepath, "w") as f:
          f.write("\n".join(str(v) for v in vals) + "\n")

        self.initializer = lookup_ops.TextFileInitializer(
            filepath, dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER,
            dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE)
      else:
        keys_tensor = constant_op.constant(
            list(range(len(vals))), dtype=dtypes.int64)
        vals_tensor = constant_op.constant(vals, dtype=dtypes.int64)
        self.initializer = lookup_ops.KeyValueTensorInitializer(
            keys_tensor, vals_tensor)

      self.table = lookup_ops.StaticHashTable(
          self.initializer, default_value=-2)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int64)])
    def use_table(self, x):
      return self.table.lookup(x)

  def verifyWorkerLocalInstance(self, coordinator, model):
    # assert capturing a worker-local resource on each worker
    for worker in coordinator._cluster.workers:
      with coordinator_context.with_dispatch_context(worker):
        captures = model.use_table.get_concrete_function().captured_inputs
        resource_capture = [t for t in captures if t.dtype == dtypes.resource]
        self.assertNotEmpty(resource_capture)
        for capture in resource_capture:
          self.assertEqual(
              capture.device,
              device_util.canonicalize("/CPU:0", default=worker.device_name))

  @combinations.generate(source_combination)
  def testInModelAndCapture(self, source):

    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")

    model = self.Model(source, file_path)
    func_captures = model.use_table.get_concrete_function(
    ).graph.external_captures
    self.assertLen(func_captures, 2)
    self.assertTrue(
        any(model.table.resource_handle is t for t in func_captures))
    deferred_captures = model.use_table.get_concrete_function(
    ).graph.deferred_external_captures
    self.assertEmpty(deferred_captures)

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy)
    with strategy.scope():
      distributed_model = self.Model("value", file_path)
    func_captures = distributed_model.use_table.get_concrete_function(
    ).graph.external_captures
    # One less external_capture, since the table handle becomes a closure in the
    # deferred_external_capture
    self.assertLen(func_captures, 1)
    self.assertFalse(
        any(model.table.resource_handle is t for t in func_captures))
    deferred_captures = distributed_model.use_table.get_concrete_function(
    ).graph.deferred_external_captures
    self.assertNotEmpty(deferred_captures)

    self.verifyWorkerLocalInstance(coordinator, distributed_model)

  @combinations.generate(source_combination)
  def testLookupInNestedTFWhileLoop(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy=strategy)

    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)

    @def_function.function
    def replica_fn(batch_data):
      replica_result = array_ops.zeros(shape=(), dtype=dtypes.int64)
      for _ in math_ops.range(10):
        replica_result += math_ops.reduce_sum(model.use_table(batch_data))
      return replica_result

    @def_function.function
    def step_fn(iterator):

      step_result = array_ops.zeros(shape=(), dtype=dtypes.int64)
      for _ in math_ops.range(10):
        step_result += strategy.run(replica_fn, args=(next(iterator),))

      return step_result

    dataset = (
        dataset_ops.DatasetV2.from_tensors(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)).repeat().batch(
                24, drop_remainder=True).prefetch(2))
    distributed_dataset = coordinator.create_per_worker_dataset(
        strategy.experimental_distribute_dataset(dataset))

    results = []
    for _ in range(10):
      results.append(
          coordinator.schedule(step_fn, args=(iter(distributed_dataset),)))

    coordinator.join()

    for r in results:
      self.assertAllClose(-2400, r.fetch())

  @combinations.generate(source_combination)
  def testDistributeTableSaveAndServe(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)

    model_dir = self.get_temp_dir()
    tf_save.save(model, model_dir)

    loaded_without_strategy = tf_load.load(model_dir)
    loaded_func_captures_without_strategy = (
        loaded_without_strategy.use_table.get_concrete_function().graph
        .external_captures)
    loaded_func_deferred_captures_without_strategy = (
        loaded_without_strategy.use_table.get_concrete_function().graph
        .deferred_external_captures)
    self.assertLen(loaded_func_captures_without_strategy, 2)
    self.assertEmpty(loaded_func_deferred_captures_without_strategy)

    self.assertAllEqual(
        loaded_without_strategy.use_table(
            constant_op.constant([0, 1, 3], dtype=dtypes.int64)), [0, 1, -2])

  @combinations.generate(source_combination)
  def testDistributeTableSaveAndLoadUnderStrategy(self, source):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        self.cluster_resolver)
    coordinator = coordinator_lib.ClusterCoordinator(strategy)
    file_path = os.path.join(self.get_temp_dir(), "text_file_initializer")
    with strategy.scope():
      model = self.Model(source, file_path)
    model_dir = self.get_temp_dir()
    tf_save.save(model, model_dir)

    with strategy.scope():
      loaded = tf_load.load(model_dir)

    loaded_func_captures = (
        loaded.use_table.get_concrete_function().graph.external_captures)
    loaded_func_deferred_captures = (
        loaded.use_table.get_concrete_function().graph
        .deferred_external_captures)
    # Compared with loading without strategy, there is one less
    # external_capture, since the captured table handle has been swapped to a
    # closure in the deferred_external_capture
    self.assertLen(loaded_func_captures, 1)
    self.assertNotEmpty(loaded_func_deferred_captures)

    self.assertIsInstance(loaded.table, ps_values.DistributedTable)

    self.assertLen([
        t for t in loaded.use_table.get_concrete_function().captured_inputs
        if t.dtype == dtypes.resource
    ], 1)

    self.verifyWorkerLocalInstance(coordinator, loaded)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
