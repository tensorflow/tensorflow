# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for parameter_server_client.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.client import client
from tensorflow.python.distribute.client import parameter_server_client
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.server_lib import ClusterSpec


def make_client(num_workers, num_ps):
  # TODO(rchao): Test the internal rpc_layer version.
  cluster_def = multi_worker_test_base.create_in_process_cluster(
      num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc")
  cluster_def["chief"] = [
      "localhost:%d" % multi_worker_test_base.pick_unused_port()
  ]
  cluster_resolver = SimpleClusterResolver(
      ClusterSpec(cluster_def), rpc_layer="grpc")
  return parameter_server_client.ParameterServerClient(cluster_resolver)


class ParameterServerClientTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ParameterServerClientTest, cls).setUpClass()
    cls.client = make_client(num_workers=3, num_ps=2)

  def testBasic(self):
    self.client._strategy.extended._variable_count = 0
    with self.client.context():
      v1 = variables.Variable(initial_value=0.0)
      v2 = variables.Variable(initial_value=1.0)
    self.assertEqual(self.client._strategy.extended._variable_count, 2)

    @def_function.function
    def worker_fn():
      v1.assign_add(0.1)
      v2.assign_sub(0.2)
      return v1.read_value() / v2.read_value()

    results = self.client.schedule(worker_fn)
    logging.info("Results of experimental_run_v2: %f",
                 self.client.fetch(results))

    self.assertAlmostEqual(v1.read_value().numpy(), 0.1, delta=1e-6)
    self.assertAlmostEqual(v2.read_value().numpy(), 0.8, delta=1e-6)

  def testFnReturnNestedValues(self):
    x = constant_op.constant(1)

    @def_function.function
    def f():
      return x + 1, (x + 2, x + 3), [x + 4], {"v": x}

    got = self.client.schedule(f)
    want = 2, (3, 4), [5], {"v": 1}
    self.assertEqual(self.client.fetch(got), want)

  def testInputFunction(self):

    def input_fn():
      return dataset_ops.DatasetV2.range(1, 2)

    with self.client.context():
      v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @def_function.function
    def worker_fn(iterator):
      x = next(iterator)
      v.assign_add(x)
      return x

    distributed_dataset = self.client.create_per_worker_dataset(input_fn)
    result = self.client.schedule(worker_fn, args=(iter(distributed_dataset),))
    result = self.client.fetch(result)
    self.assertEqual(result, (1,))
    result = self.client.schedule(worker_fn, args=(iter(distributed_dataset),))
    result = self.client.fetch(result)
    self.assertEqual(result, (1,))

    self.assertAlmostEqual(v.read_value().numpy(), 2, delta=1e-6)

  def testAsyncScheduleAndJoin(self):

    def input_fn():
      return dataset_ops.DatasetV2.from_tensor_slices([2] * 10)

    with self.client.context():
      v = variables.Variable(initial_value=0, dtype=dtypes.int32)

    # TODO(yuefengz): the following tf.function has a return value which is None
    # in its structured_outputs.
    @def_function.function
    def worker_fn(iterator):
      x = next(iterator)
      v.assign_add(x)

    distributed_dataset = self.client.create_per_worker_dataset(input_fn)

    iterator = iter(distributed_dataset)

    # Verifying joining without any scheduling doesn't hang.
    self.client.join()
    self.assertEqual(v.read_value().numpy(), 0)

    for _ in range(5):
      self.client.schedule(worker_fn, args=(iterator,))
    self.client.join()

    # With 5 addition it should be 2*5 = 10.
    self.assertEqual(v.read_value().numpy(), 10)

    for _ in range(5):
      self.client.schedule(worker_fn, args=(iterator,))

    # Verifying multiple join is fine.
    self.client.join()
    self.client.join()
    self.client.join()

    self.assertTrue(self.client.done())

    # Likewise, it's now 20.
    self.assertEqual(v.read_value().numpy(), 20)

  def testInputFunctionWithMap(self):
    self._map_fn_tracing_count = 0

    def input_fn():
      def map_fn(x):
        self._map_fn_tracing_count += 1
        return x + 10
      return dataset_ops.DatasetV2.range(0, 10).map(map_fn)

    @def_function.function
    def worker_fn(iterator):
      return next(iterator)

    distributed_dataset = (
        self.client.create_per_worker_dataset(input_fn))
    result = self.client.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    self.assertEqual(result.fetch(), (10,))
    self.assertEqual(self._map_fn_tracing_count, 1)

  def testInputFunctionCreateVariables(self):

    def input_fn():
      v = variables.Variable(initial_value=0.0)
      return v.read_value()

    with self.assertRaises(ValueError):
      self.client.create_per_worker_dataset(input_fn)


class LimitedClosureQueueSizeBasicTest(ParameterServerClientTest):
  """Test basic functionality works with explicit maximum closure queue size.

  Execute the same set of test cases as in ParameterServerClientTest, with an
  explicit size limit for the closure queue. Note that even when the queue size
  is set to infinite, there is still a maximum practical size (depends on host
  memory limit) that might cause the queue.put operations to be blocking when
  scheduling a large number of closures on a big cluster. These tests make sure
  that the client does not run into deadlocks in such scenario.
  """

  @classmethod
  def setUpClass(cls):
    super(LimitedClosureQueueSizeBasicTest, cls).setUpClass()
    client._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.client = make_client(num_workers=3, num_ps=2)


class VariablePartitioningScopeTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(VariablePartitioningScopeTest, cls).setUpClass()
    cls.client = make_client(num_workers=3, num_ps=2)

  def testBasic(self):
    with self.client.context():
      with self.client.experimental_variable_partitioning_scope():
        init1 = init_ops_v2.Constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        v1 = variables.Variable(
            initial_value=lambda: init1(shape=(5, 2), dtype=dtypes.int64),
            shape=(5, 2),
            dtype=dtypes.int64)

        init2 = init_ops_v2.Constant([0, 1, 2, 3, 4, 5])
        v2 = variables.Variable(
            initial_value=lambda: init2(shape=(6, 1), dtype=dtypes.int64),
            shape=(6, 1),
            dtype=dtypes.int64)

    self.assertIsInstance(v1, sharded_variable.ShardedVariable)
    self.assertLen(v1.variables, 2)
    self.assertRegex(v1.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v1.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v1.variables[0].read_value().numpy(),
                        [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(v1.variables[1].read_value().numpy(), [[6, 7], [8, 9]])

    self.assertIsInstance(v2, sharded_variable.ShardedVariable)
    self.assertLen(v2.variables, 2)
    self.assertRegex(v2.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertRegex(v2.variables[1].device, "/job:ps/replica:0/task:1")
    self.assertAllEqual(v2.variables[0].read_value().numpy(), [[0], [1], [2]])
    self.assertAllEqual(v2.variables[1].read_value().numpy(), [[3], [4], [5]])

  def testSurplusPS(self):
    with self.client.context():
      with self.client.experimental_variable_partitioning_scope():
        initializer = init_ops_v2.Constant([0])

        v = variables.Variable(
            initial_value=lambda: initializer(shape=(1,), dtype=dtypes.int64),
            shape=(1,),
            dtype=dtypes.int64)

    self.assertIsInstance(v, sharded_variable.ShardedVariable)
    self.assertLen(v.variables, 1)
    self.assertRegex(v.variables[0].device, "/job:ps/replica:0/task:0")
    self.assertAllEqual(v.variables[0].read_value().numpy(), [0])

  def testInvalidArgument(self):
    with self.assertRaisesRegex(ValueError, "initial_value"):
      with self.client.experimental_variable_partitioning_scope():
        variables.Variable(initial_value=[0, 1, 2], shape=(3,))

    with self.assertRaisesRegex(ValueError, "shape"):
      with self.client.experimental_variable_partitioning_scope():
        initializer = init_ops_v2.Constant([0, 1, 2])
        variables.Variable(
            initial_value=lambda: initializer(shape=(3,), dtype=dtypes.int64),
            dtype=dtypes.int64)

  def testPerWorkerValue(self):
    var_shape = tuple()
    var_dtype = dtypes.float32
    var_name = "var"

    def create_var():
      var = variables.Variable(
          initial_value=0.0, dtype=var_dtype, name=var_name)
      self.assertIn("worker", var.device)
      return var

    worker_local_var = self.client._create_per_worker_resources(create_var)

    # The following is a workaround to allow `worker_local_var` to be passed in
    # as args to the `client.schedule` method which requires tensor specs to
    # trace tf.function but _create_worker_resources' return values don't have
    # tensor specs. We can get rid of this workaround once
    # _create_worker_resources is able to infer the tensor spec of the return
    # value of the function passed in. See b/154675763.
    for var in worker_local_var._values:
      var._set_type_spec(tensor_spec.TensorSpec(var_shape, var_dtype, var_name))

    def worker_fn(var):
      var.assign_add(1.0)

    for _ in range(10):
      # Which slice of `worker_local_var` will be used will depend on which
      # worker the `worker_fn` gets scheduled on.
      self.client.schedule(worker_fn, args=(worker_local_var,))
    self.client.join()

    var_sum = sum(self.client.fetch(worker_local_var._values))
    self.assertEqual(var_sum, 10.0)


class ErrorReportingTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ErrorReportingTest, cls).setUpClass()
    cls.client = make_client(num_workers=3, num_ps=2)

    with cls.client.context():
      cls.iteration = variables.Variable(initial_value=0.0)

  @def_function.function
  def _normal_function(self):
    x = random_ops.random_uniform((2, 10))
    y = random_ops.random_uniform((10, 2))
    self.iteration.assign_add(1.0)
    return math_ops.reduce_mean(math_ops.matmul(x, y))

  @def_function.function
  def _error_function(self):
    x = random_ops.random_uniform((2, 10))
    y = random_ops.random_uniform((10, 2))
    check_ops.assert_non_positive_v2(math_ops.reduce_sum(math_ops.matmul(x, y)))
    self.iteration.assign_add(1.0)
    return self.iteration

  def testJoinRaiseError(self):
    for _ in range(3):
      self.client.schedule(self._normal_function)
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

  def testScheduleRaiseError(self):
    for _ in range(3):
      self.client.schedule(self._normal_function)
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      while True:
        self.client.schedule(self._normal_function)

  def testErrorWillbeCleared(self):
    self.skipTest("b/157597579")
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

    for _ in range(3):
      self.client.schedule(self._normal_function)
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

  def testFutureReturnError(self):
    result = self.client.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      result.fetch()

    # Clear the error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

  def testInputError(self):
    aborted = self.client.schedule(self._error_function)

    @def_function.function
    def func(x):
      return x + 1.0

    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

    result = self.client.schedule(func, args=(aborted,))
    with self.assertRaises(client.InputError):
      result.fetch()

    with self.assertRaises(client.InputError):
      self.client.join()


class LimitedClosureQueueErrorTest(ErrorReportingTest):
  """Test error reporting works with explicit maximum closure queue size.

  Execute the same set of test cases as in ErrorReportingTest, with an explicit
  size limit for the closure queue.
  """

  @classmethod
  def setUpClass(cls):
    super(LimitedClosureQueueErrorTest, cls).setUpClass()
    client._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.client = make_client(num_workers=3, num_ps=2)

    with cls.client.context():
      cls.iteration = variables.Variable(initial_value=0.0)


if __name__ == "__main__":
  test.main()
