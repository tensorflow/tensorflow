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
"""Tests for `Client` when used together with `ParameterServerStrategyV2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import threading
from absl import logging

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.client import client as client_lib
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.server_lib import ClusterSpec


class ErrorReportingThread(threading.Thread):

  error = None

  def __init__(self, *args, **kwargs):
    assert "target" in kwargs
    target = kwargs["target"]

    @functools.wraps(target)
    def wrapped_target(*args, **kwargs):
      try:
        return target(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-except
        ErrorReportingThread.error = e

    kwargs["target"] = wrapped_target
    super(ErrorReportingThread, self).__init__(*args, **kwargs)


class TestCaseWithErrorReportingThread(test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._threading_thread = threading.Thread
    threading.Thread = ErrorReportingThread
    super(TestCaseWithErrorReportingThread, cls).setUpClass()

  @classmethod
  def tearDownClass(cls):
    super(TestCaseWithErrorReportingThread, cls).tearDownClass()
    threading.Thread = cls._threading_thread

  def setUp(self):
    ErrorReportingThread.error = None
    super(TestCaseWithErrorReportingThread, self).setUp()

  def tearDown(self):
    super(TestCaseWithErrorReportingThread, self).tearDown()
    if ErrorReportingThread.error:
      raise ErrorReportingThread.error  # pylint: disable=raising-bad-type


def make_client(num_workers, num_ps):
  # TODO(rchao): Test the internal rpc_layer version.
  cluster_def = multi_worker_test_base.create_in_process_cluster(
      num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc")
  cluster_def["chief"] = [
      "localhost:%d" % multi_worker_test_base.pick_unused_port()
  ]
  cluster_resolver = SimpleClusterResolver(
      ClusterSpec(cluster_def), rpc_layer="grpc")
  strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
      cluster_resolver)
  return client_lib.Client(strategy)


class ParameterServerClientTest(TestCaseWithErrorReportingThread):

  @classmethod
  def setUpClass(cls):
    super(ParameterServerClientTest, cls).setUpClass()
    cls.client = make_client(num_workers=3, num_ps=2)
    cls.strategy = cls.client.strategy

  def testBasic(self):
    self.strategy.extended._variable_count = 0
    with self.strategy.scope():
      v1 = variables.Variable(initial_value=0.0)
      v2 = variables.Variable(initial_value=1.0)
    self.assertEqual(self.strategy.extended._variable_count, 2)

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

    with self.strategy.scope():
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

    with self.strategy.scope():
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

  def testDisallowRemoteValueAsInput(self):

    @def_function.function
    def func_0():
      return 1.0

    @def_function.function
    def func_1(x):
      return x + 1.0

    remote_v = self.client.schedule(func_0)
    with self.assertRaises(ValueError):
      self.client.schedule(func_1, args=(remote_v,))


class LimitedClosureQueueSizeBasicTest(ParameterServerClientTest):
  """Test basic functionality works with explicit maximum closure queue size.

  Execute the same set of test cases as in `ParameterServerClientTest`, with an
  explicit size limit for the closure queue. Note that even when the queue size
  is set to infinite, there is still a maximum practical size (depends on host
  memory limit) that might cause the queue.put operations to be blocking when
  scheduling a large number of closures on a big cluster. These tests make sure
  that the client does not run into deadlocks in such scenario.
  """

  @classmethod
  def setUpClass(cls):
    super(LimitedClosureQueueSizeBasicTest, cls).setUpClass()
    client_lib._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.client = make_client(num_workers=3, num_ps=2)
    cls.strategy = cls.client.strategy


class ErrorReportingTest(TestCaseWithErrorReportingThread):

  @classmethod
  def setUpClass(cls):
    super(ErrorReportingTest, cls).setUpClass()
    cls.client = make_client(num_workers=3, num_ps=2)
    cls.strategy = cls.client.strategy

    with cls.strategy.scope():
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

  @def_function.function
  def _long_function(self):
    x = random_ops.random_uniform((1000, 1000))
    for _ in math_ops.range(10000):
      a = random_ops.random_uniform((1000, 1000))
      b = random_ops.random_uniform((1000, 1000))
      x += math_ops.matmul(a, b)
    return x

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

  def testScheduleRaiseErrorWithMultipleFailure(self):
    for _ in range(3):
      self.client.schedule(self._normal_function)
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      while True:
        self.client.schedule(self._error_function)
    self.client.join()

  def testErrorWillbeCleared(self):
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

    for _ in range(3):
      self.client.schedule(self._normal_function)
    self.client.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

  def testRemoteValueReturnError(self):
    result = self.client.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      result.fetch()

    # Clear the error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

  def testInputError(self):

    worker_local_val = self.client._create_per_worker_resources(
        self._error_function)

    @def_function.function
    def func(x):
      return x + 1

    result = self.client.schedule(func, args=(worker_local_val,))
    with self.assertRaises(client_lib.InputError):
      self.client.join()

    with self.assertRaises(client_lib.InputError):
      result.fetch()

  def testCancellation(self):
    for _ in range(3):
      self.client.schedule(self._normal_function)
    long_function = self.client.schedule(self._long_function)
    self.client.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      self.client.join()

    with self.assertRaises(client_lib.FunctionRetryableError):
      long_function.fetch()

    for _ in range(3):
      self.client.schedule(self._normal_function)
    self.client.join()


class LimitedClosureQueueErrorTest(ErrorReportingTest):
  """Test error reporting works with explicit maximum closure queue size.

  Execute the same set of test cases as in ErrorReportingTest, with an explicit
  size limit for the closure queue.
  """

  @classmethod
  def setUpClass(cls):
    super(LimitedClosureQueueErrorTest, cls).setUpClass()
    client_lib._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.client = make_client(num_workers=3, num_ps=2)
    cls.strategy = cls.client.strategy

    with cls.client.strategy.scope():
      cls.iteration = variables.Variable(initial_value=0.0)


class StrategyRunTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(StrategyRunTest, cls).setUpClass()
    cls.client = make_client(num_workers=1, num_ps=1)
    cls.strategy = cls.client.strategy

  def testStrategyRun(self):
    self.assertFalse(distribution_strategy_context.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribution_strategy_context.in_cross_replica_context())
      v = variables.Variable(initial_value=1)

      @def_function.function
      def worker_fn(input_tensor):

        def replica_fn(input_tensor):
          # Within `replica_fn`, it has to be in a replica context.
          self.assertFalse(
              distribution_strategy_context.in_cross_replica_context())
          return input_tensor + v

        return self.strategy.run(replica_fn, args=(input_tensor,))

      # Asserting scheduling in scope has the expected behavior.
      result = self.client.schedule(worker_fn, args=(constant_op.constant(3),))
      self.assertIsInstance(result, client_lib.RemoteValue)
      self.assertEqual(result.fetch(), 4)

    # Asserting scheduling out of scope has the expected behavior.
    result = self.client.schedule(worker_fn, args=(constant_op.constant(3),))
    self.assertEqual(result.fetch(), 4)


if __name__ == "__main__":
  test.main()
