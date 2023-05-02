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
"""Tests for coordinator.py."""

import collections
import contextlib
import functools
import gc
import os
import platform
import sys
import threading
import time
import traceback
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training.server_lib import ClusterSpec


class ClosureWithOutput(coordinator_lib.Closure):

  def __init__(self, function, cancellation_mgr=None, args=None, kwargs=None):
    super(ClosureWithOutput, self).__init__(
        function, cancellation_mgr=cancellation_mgr, args=args, kwargs=kwargs)
    self.output_remote_value = self.build_output_remote_value()


class CoordinatedClosureQueueTest(test.TestCase):

  def testBasic(self):
    queue = coordinator_lib._CoordinatedClosureQueue()
    closure1 = self._create_closure(queue._cancellation_mgr)
    queue.put(closure1)
    self.assertIs(closure1, queue.get())
    self.assertFalse(queue.done())
    queue.put_back(closure1)
    self.assertEqual(closure1, queue.get())
    queue.mark_finished()
    self.assertTrue(queue.done())
    queue.wait()

  def testProcessAtLeaseOnce(self):
    closure_queue = coordinator_lib._CoordinatedClosureQueue()
    labels = ['A', 'B', 'C', 'D', 'E']
    processed_count = collections.defaultdict(int)

    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def process_queue():
      with coord.stop_on_exception():
        has_been_put_back = False
        while True:
          closure = closure_queue.get(timeout=30)
          if closure is None:
            break
          if not has_been_put_back:
            has_been_put_back = True
            closure_queue.put_back(closure)
            continue
          closure._function()
          closure_queue.mark_finished()

    def get_func(label):

      def func():
        time.sleep(3)
        processed_count[label] += 1

      return func

    cm = cancellation.CancellationManager()
    for label in labels:
      closure_queue.put(ClosureWithOutput(get_func(label), cm))
    t1 = threading.Thread(target=process_queue, daemon=True)
    t1.start()
    t2 = threading.Thread(target=process_queue, daemon=True)
    t2.start()

    # Make sure multiple wait() calls are fine.
    closure_queue.wait()
    closure_queue.wait()
    closure_queue.wait()
    closure_queue.wait()

    self.assertEqual(processed_count, collections.Counter(labels))

    coord.join([t1, t2])

  def testNotifyBeforeWait(self):
    closure_queue = coordinator_lib._CoordinatedClosureQueue()

    def func():
      logging.info('func running')

    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def process_queue():
      with coord.stop_on_exception():
        closure_queue.get()
        closure_queue.mark_finished()

    closure_queue.put(ClosureWithOutput(func, closure_queue._cancellation_mgr))
    t = threading.Thread(target=process_queue)
    t.start()
    coord.join([t])

    # This test asserts that waiting at the time the function has been processed
    # doesn't time out.
    closure_queue.wait()

  def _assert_one_unblock_the_other(self, first_fn, second_fn):
    """Asserts `second_fn` wouldn't return before `first_fn` is finished."""
    first_fn_done = threading.Event()
    second_fn_done = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def wrapped_first_fn():
      with coord.stop_on_exception():
        self.assertFalse(second_fn_done.is_set())
        first_fn()
        first_fn_done.set()

    self.assertFalse(first_fn_done.is_set())
    t = threading.Thread(target=wrapped_first_fn)
    t.start()

    second_fn()
    self.assertTrue(first_fn_done.is_set())
    second_fn_done.set()

    coord.join([t])

  def _run_two_fns_in_parallel(self, first_fn, second_fn):
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def wrapped_first_fn():
      with coord.stop_on_exception():
        first_fn()

    t = threading.Thread(target=wrapped_first_fn)
    t.start()

    second_fn()
    coord.join([t])

  def testWaitRaiseErrorAfterMarkFailure(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue = coordinator_lib._CoordinatedClosureQueue()
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    closure = closure_queue.get()

    wait_finish_event = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    # Using a thread to verify that closure_queue.wait() will not return until
    # all inflight closures are finished.

    def mark_finished_fn():
      try:
        raise ValueError('Some error.')
      except ValueError as e:
        closure_queue.mark_failed(e)

    def wait_fn():
      with self.assertRaises(ValueError):
        closure_queue.wait()

    self._assert_one_unblock_the_other(mark_finished_fn, wait_fn)

    self.assertTrue(closure_queue.done())

  def _create_closure(self, cancellation_mgr):

    @def_function.function()
    def some_function():
      return 1.0

    return ClosureWithOutput(some_function, cancellation_mgr)

  def _put_two_closures_and_get_one(self):
    closure_queue = coordinator_lib._CoordinatedClosureQueue()
    closure1 = self._create_closure(closure_queue._cancellation_mgr)
    closure_queue.put(closure1)

    closure2 = self._create_closure(closure_queue._cancellation_mgr)
    closure_queue.put(closure2)

    closure_got = closure_queue.get()  # returns closure1
    self.assertIs(closure_got, closure1)
    self.assertIsNot(closure_got, closure2)
    return closure_queue, closure1, closure2

  def testPutRaiseError(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError())

    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        errors.CancelledError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2.output_remote_value.fetch()

    # The error is cleared.
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

  def testWaitRaiseError(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError())

    with self.assertRaises(ValueError):
      closure_queue.wait()
    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        errors.CancelledError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2.output_remote_value.fetch()

    # The error is cleared.
    closure_queue.wait()

  def testDoneRaiseError(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, _ = self._put_two_closures_and_get_one()

    self.assertFalse(closure_queue.done())
    closure_queue.mark_failed(ValueError())
    with self.assertRaises(ValueError):
      closure_queue.done()

  def _set_error(self, closure_queue, closure, error):
    try:
      raise error
    except Exception as e:  # pylint: disable=broad-except
      closure.output_remote_value._set_error(e)
      closure_queue.mark_failed(e)

  def _test_cancel_closure_when_error(self, call_wait):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    closure_queue.get()
    # At this moment, there are two inflight, one in queue.
    self.assertEqual(closure_queue._inflight_closure_count, 2)

    # Hold a copy of the queue's cancellation manager at this point
    initial_cm = closure_queue._cancellation_mgr

    # Simulating closure1 fails.
    self._set_error(closure_queue, closure1, ValueError('Some error.'))

    # At this moment, there are one inflight, one in queue.
    self.assertEqual(closure_queue._queue.qsize(), 1)
    self.assertEqual(closure_queue._inflight_closure_count, 1)

    closure3 = self._create_closure(closure_queue._cancellation_mgr)

    def fake_cancellation():
      self._set_error(closure_queue, closure2,
                      ValueError('Fake cancellation error.'))

    def report_error():
      # It should not report the fake cancellation error.
      with self.assertRaisesRegex(ValueError, 'Some error.'):
        # Verifying `wait()` or `put()` raises even if one closure is in
        # flight.
        if call_wait:
          closure_queue.wait()
        else:
          closure_queue.put(closure3)

    self._assert_one_unblock_the_other(fake_cancellation, report_error)

    # The original cancellation manager of the queue has been cancelled.
    self.assertTrue(initial_cm.is_cancelled)

    # At this moment, there is zero inflight, nothing in queue.
    self.assertTrue(closure_queue._queue.empty())
    self.assertEqual(closure_queue._inflight_closure_count, 0)
    self.assertIsNone(closure_queue._error)

    # This asserts that closure1 has errored.
    with self.assertRaisesRegex(ValueError, 'Some error.'):
      closure1.output_remote_value.fetch()

    # The following asserts that closure3 should have been cancelled.
    if not call_wait:
      with self.assertRaisesRegex(
          errors.CancelledError,
          'The corresponding function is cancelled. Please reschedule the '
          'function.'):
        closure3.output_remote_value.fetch()

    # Closure2 was an inflight closure when it got cancelled.
    self.assertEqual(closure2.output_remote_value._status,
                     remote_value.RemoteValueStatus.READY)
    with self.assertRaisesRegex(ValueError, 'Fake cancellation error.'):
      closure2.output_remote_value.fetch()

    # This asserts that the queue has a clear state.
    self.testBasic()

  def testWaitRaiseErrorAfterCancelClosure(self):
    self._test_cancel_closure_when_error(call_wait=True)

  def testPutRaiseErrorAfterCancelClosure(self):
    self._test_cancel_closure_when_error(call_wait=False)

  def testStateIsRestoredAfterJoinIsCalled(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, _ = self._put_two_closures_and_get_one()
    self.assertEqual(closure_queue._inflight_closure_count, 1)
    closure_queue.mark_failed(ValueError('test error'))
    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

    # Its error should have been cleared.
    self.assertIsNone(closure_queue._error)
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    self.assertIsNone(closure_queue._error)

  def testThreadSafey(self):
    thread_count = 10
    queue = coordinator_lib._CoordinatedClosureQueue()

    # Each thread performs 20 queue actions: 10 are `put_back` and 10 are
    # `mark_finished`.
    action_count = 20

    def func():
      for i in range(action_count):
        closure = queue.get()
        if i % 2 == 0:
          queue.put_back(closure)
        else:
          queue.mark_finished()

    threads = [threading.Thread(target=func) for i in range(thread_count)]
    for t in threads:
      t.start()

    for _ in range(thread_count * action_count // 2):
      queue.put(self._create_closure(queue._cancellation_mgr))
    queue.wait()
    self.assertTrue(queue.done())

  def testPutGetWithTag(self):
    queue = coordinator_lib._CoordinatedClosureQueue()

    closure1 = self._create_closure(queue._cancellation_mgr)
    closure2 = self._create_closure(queue._cancellation_mgr)
    closure3 = self._create_closure(queue._cancellation_mgr)

    def put_fn():
      queue.put(closure3, tag=1)
      queue.put(closure2, tag=2)
      queue.put(closure1)

    def get_fn():
      # The get should only return the closure with tag 2.
      self.assertIs(closure2, queue.get(tag=2))

    self._run_two_fns_in_parallel(put_fn, get_fn)

    self.assertFalse(queue.done())
    self.assertEqual(closure1, queue.get())
    queue.mark_finished()

    # It will not wait for closure3
    self.assertTrue(queue.done())
    queue.wait()


class ErrorReportingThread(threading.Thread):

  error = None

  def __init__(self, *args, **kwargs):
    assert 'target' in kwargs
    target = kwargs['target']

    @functools.wraps(target)
    def wrapped_target(*args, **kwargs):
      try:
        return target(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-except
        traceback.print_exception(*sys.exc_info())
        ErrorReportingThread.error = e

    kwargs['target'] = wrapped_target
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


def make_coordinator(num_workers, num_ps):
  # TODO(rchao): Test the internal rpc_layer version.
  cluster_def = multi_worker_test_base.create_in_process_cluster(
      num_workers=num_workers, num_ps=num_ps, rpc_layer='grpc')
  cluster_def['chief'] = [
      'localhost:%d' % test_util.pick_unused_port()
  ]
  cluster_resolver = SimpleClusterResolver(
      ClusterSpec(cluster_def), rpc_layer='grpc')
  strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
      cluster_resolver)
  return coordinator_lib.ClusterCoordinator(strategy)


class CoordinatorContextTest(test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(CoordinatorContextTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=5, num_ps=2)
    cls.strategy = cls.coordinator.strategy

  def testWorkerIndexDatasetFn(self):
    def dataset_fn(context):
      del context
      dataset = dataset_ops.DatasetV2.range(10)
      worker_index = coordinator_context.get_current_worker_index()
      dataset = dataset.shard(
          num_shards=self.strategy._extended._num_workers,
          index=worker_index,
      )
      return dataset

    @def_function.function
    def per_worker_dataset_fn():
      return self.strategy.distribute_datasets_from_function(dataset_fn)

    @def_function.function
    def train_fn(iterator):
      total = constant_op.constant(0, dtype=dtypes.int64)
      for batch in iterator:
        total += math_ops.reduce_sum(batch)
      return total

    per_worker_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    with self.strategy.scope():
      iterator = iter(per_worker_dataset)
      ret_vals = []
      # Use private APIs to schedule in tagged queues to ensure each worker
      # executes only one closure.
      for ix in range(5):
        closure = coordinator_lib.Closure(
            train_fn,
            self.coordinator._cluster.closure_queue._cancellation_mgr,
            args=(iterator,))
        ret = closure.build_output_remote_value()
        # The queue doesn't keep track of tagged closures as inflight by
        # default, so hack around this for the test.
        self.coordinator._cluster.closure_queue._inflight_closure_count += 1
        self.coordinator._cluster.closure_queue.put(closure, tag=ix)
        ret_vals.append(ret)
    self.coordinator.join()

    fetched_vals = [rv.fetch() for rv in ret_vals]
    expected_results = [5, 7, 9, 11, 13]
    self.assertAllClose(sorted(fetched_vals), expected_results)


class ClusterCoordinatorTest(
    TestCaseWithErrorReportingThread, parameterized.TestCase
):

  @classmethod
  def setUpClass(cls):
    super(ClusterCoordinatorTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=5, num_ps=2)
    cls.strategy = cls.coordinator.strategy

  def testClusterCoordinatorOnlyInitOnce(self):
    cluster = self.coordinator._cluster
    same_coordinator = coordinator_lib.ClusterCoordinator(self.strategy)
    self.assertIs(self.coordinator, same_coordinator)
    self.assertIs(cluster, same_coordinator._cluster)

  def testFnReturnNestedValues(self):
    x = constant_op.constant(1)

    @def_function.function
    def f():
      return x + 1, (x + 2, x + 3), [x + 4], {'v': x}

    got = self.coordinator.schedule(f)
    want = 2, (3, 4), [5], {'v': 1}
    self.assertEqual(got.fetch(), want)
    self.assertEqual(self.coordinator.fetch(got), want)

  def testFetchingRemoteValueStructure(self):
    self.skipTest('b/171040359: flaky test')
    x = constant_op.constant(1)

    @def_function.function
    def f():
      return x + 1, (x + 2, x + 3), [x + 4], {'v': x}

    want = 2, (3, 4), [5], {'v': 1}
    remote_value_list = [self.coordinator.schedule(f) for _ in range(5)]
    self.assertAllEqual(
        self.coordinator.fetch(remote_value_list), [want for _ in range(5)])

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

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    result = self.coordinator.fetch(result)
    self.assertEqual(result, (1,))
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    result = self.coordinator.fetch(result)

    self.assertEqual(result, (1,))
    self.assertAlmostEqual(v.read_value(), 2, delta=1e-6)

  def testAsyncScheduleAndJoin(self):
    if test_util.is_xla_enabled():
      self.skipTest('Assign_add is not deterministic across threads in XLA')

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

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)

    iterator = iter(distributed_dataset)

    # Verifying joining without any scheduling doesn't hang.
    self.coordinator.join()
    self.assertEqual(v.read_value().numpy(), 0)

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))
    self.coordinator.join()

    # With 5 addition it should be 2*5 = 10.
    self.assertEqual(v.read_value().numpy(), 10)

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))

    # Verifying multiple join is fine.
    self.coordinator.join()
    self.coordinator.join()
    self.coordinator.join()

    self.assertTrue(self.coordinator.done())

    # Likewise, it's now 20.
    self.assertEqual(v.read_value().numpy(), 20.)

  @parameterized.parameters(True, False)
  def testInputFunctionWithMap(self, use_input_fn):
    self._map_fn_tracing_count = 0

    def input_fn():

      def map_fn(x):
        self._map_fn_tracing_count += 1
        return x + 10

      return dataset_ops.DatasetV2.range(0, 10).map(map_fn)

    @def_function.function
    def worker_fn(iterator):
      return next(iterator)

    if use_input_fn:
      distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    else:
      distributed_dataset = self.coordinator.create_per_worker_dataset(
          input_fn())

    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    self.assertEqual(result.fetch(), (10,))
    self.assertEqual(self._map_fn_tracing_count, 1)

  def testInputFunctionCreateVariables(self):

    def input_fn():
      v = variables.Variable(initial_value=0.0)
      return v.read_value()

    with self.assertRaises(ValueError):
      self.coordinator.create_per_worker_dataset(input_fn)

  @parameterized.parameters(True, False)
  def testDatasetsShuffledDifferently(self, use_input_fn):
    # This test requires at least two workers in the cluster.
    self.assertGreaterEqual(len(self.coordinator._cluster.workers), 2)

    random_seed.set_random_seed(None)

    def input_fn():
      dataset = dataset_ops.DatasetV2.range(0, 100).shuffle(100).batch(1)
      return self.strategy.experimental_distribute_dataset(dataset)

    if use_input_fn:
      distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    else:
      distributed_dataset = self.coordinator.create_per_worker_dataset(
          input_fn())
    distributed_iterator = iter(distributed_dataset)
    # Get elements from the first two iterators.
    iterator_1 = distributed_iterator._values[0]
    iterator_1 = iterator_1.fetch()
    elements_in_iterator_1 = [
        self.strategy.experimental_local_results(e)
        for e in iterator_1
    ]
    iterator_2 = distributed_iterator._values[1]
    iterator_2 = iterator_2.fetch()
    elements_in_iterator_2 = [
        self.strategy.experimental_local_results(e)
        for e in iterator_2
    ]

    self.assertNotAllEqual(elements_in_iterator_1, elements_in_iterator_2)

  def testPerWorkerValue(self):
    var_shape = tuple()
    var_dtype = dtypes.float32
    var_name = 'var'

    # This should not be a tf.function, as variable creation is prohibited in
    # tf.functions in general.
    def create_var():
      var = variables.Variable(
          initial_value=0.0, dtype=var_dtype, name=var_name)
      self.assertIn('worker', var.device)
      return var

    worker_local_var = self.coordinator._create_per_worker_resources(create_var)
    # The following is a workaround to allow `worker_local_var` to be passed in
    # as args to the `coordinator.schedule` method which requires specs
    # to trace a tf.function, but _create_worker_resources' return values don't
    # have specs when its input function is not a tf.function. We can get rid of
    # this workaround once _create_worker_resources is able to infer the tensor
    # spec of the return value of the (non-tf) function passed in. See
    # b/154675763.
    for var in worker_local_var._values:
      var._type_spec = resource_variable_ops.VariableSpec(var_shape, var_dtype)

    @def_function.function
    def worker_fn(var):
      var.assign_add(1.0)

    for _ in range(10):
      # Which slice of `worker_local_var` will be used will depend on which
      # worker the `worker_fn` gets scheduled on.
      self.coordinator.schedule(worker_fn, args=(worker_local_var,))
    self.coordinator.join()

    var_sum = sum(self.coordinator.fetch(worker_local_var._values))
    self.assertEqual(var_sum, 10.0)

  def testDisallowRemoteValueAsInput(self):

    @def_function.function
    def func_0():
      return 1.0

    @def_function.function
    def func_1(x):
      return x + 1.0

    remote_v = self.coordinator.schedule(func_0)
    with self.assertRaises(ValueError):
      self.coordinator.schedule(func_1, args=(remote_v,))

  def testPythonFunctionNotAllowedToSchedule(self):

    def func(a):
      return array_ops.identity(a)

    with self.assertRaisesRegexp(
        TypeError,
        '`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` '
        'only accepts a `tf.function` or a concrete function.'):
      self.coordinator.schedule(func, args=(1,))

  def testDatasetPartiallyCreatedOnCoordinator(self):
    dataset = dataset_ops.DatasetV2.range(1, 10)

    @def_function.function
    def input_fn():
      return dataset.shuffle(9)

    @def_function.function
    def worker_fn(iterator):
      x = next(iterator)
      return x

    per_worker_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    self.coordinator.schedule(worker_fn, args=(iter(per_worker_dataset),))

    with self.assertRaisesRegexp(
        coordinator_lib.ClosureInputError,
        'error message is Failed copying input tensor from'):
      self.coordinator.join()

  def testPassDatasetToCreatePerWorkerDataset(self):
    dataset = dataset_ops.DatasetV2.range(1, 11).batch(4)

    @def_function.function
    def worker_fn(iterator):
      return next(iterator)

    per_worker_dataset = self.coordinator.create_per_worker_dataset(dataset)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(per_worker_dataset),))
    result = result.fetch()
    expected_result = math_ops.range(1., 5.)

    self.assertAllEqual(result, (expected_result))

  def testMultipleDatasets(self):

    def input_fn1():
      return dataset_ops.DatasetV2.range(0, 5)

    def input_fn2():
      return dataset_ops.DatasetV2.range(5, 10)

    per_worker_dataset1 = self.coordinator.create_per_worker_dataset(input_fn1)
    per_worker_iterator1 = iter(per_worker_dataset1)
    per_worker_dataset2 = self.coordinator.create_per_worker_dataset(input_fn2)
    per_worker_iterator2 = iter(per_worker_dataset2)

    @def_function.function
    def worker_fn(iterator1, iterator2):
      return next(iterator1) + next(iterator2)

    result = self.coordinator.schedule(
        worker_fn, args=(per_worker_iterator1, per_worker_iterator2))
    self.assertEqual(result.fetch(), 5.0)

    per_worker_dataset3 = self.coordinator.create_per_worker_dataset(input_fn1)
    per_worker_iterator3 = iter(per_worker_dataset3)

    result = self.coordinator.schedule(
        worker_fn, args=(per_worker_iterator3, per_worker_iterator2))
    self.assertGreaterEqual(result.fetch(), 5.0)

  def testRepeatedIteratorCreation(self):

    def input_fn():
      return dataset_ops.DatasetV2.range(1, 100)

    per_worker_dataset1 = self.coordinator.create_per_worker_dataset(input_fn)
    per_worker_dataset2 = self.coordinator.create_per_worker_dataset(input_fn)

    @def_function.function
    def worker_fn(iterator1, iterator2):
      return next(iterator1) + next(iterator2)

    for _ in range(10):
      per_worker_iterator1 = iter(per_worker_dataset1)
      per_worker_iterator2 = iter(per_worker_dataset2)
      result = self.coordinator.schedule(
          worker_fn, args=(per_worker_iterator1, per_worker_iterator2))
      for _ in range(10):
        self.coordinator.schedule(
            worker_fn, args=(per_worker_iterator1, per_worker_iterator2))
      self.coordinator.join()
      self.assertGreaterEqual(result.fetch(), 2.0)
    del per_worker_iterator1, per_worker_iterator2
    gc.collect()

    # There shouldn't be any live iterator objects.
    for w in self.coordinator._cluster.workers:
      for r in w._resource_remote_value_refs:
        self.assertIsNone(r())


class LimitedClosureQueueSizeBasicTest(ClusterCoordinatorTest):
  """Test basic functionality works with explicit maximum closure queue size.

  Execute the same set of test cases as in `ClusterCoordinatorTest`, with an
  explicit size limit for the closure queue. Note that even when the queue size
  is set to infinite, there is still a maximum practical size (depends on host
  memory limit) that might cause the queue.put operations to be blocking when
  scheduling a large number of closures on a big cluster. These tests make sure
  that the coordinator does not run into deadlocks in such scenario.
  """

  @classmethod
  def setUpClass(cls):
    super(LimitedClosureQueueSizeBasicTest, cls).setUpClass()
    coordinator_lib._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.coordinator = make_coordinator(num_workers=5, num_ps=2)
    cls.strategy = cls.coordinator.strategy


class ScheduleStartDelayTest(ClusterCoordinatorTest):
  """Test basic functionality works with worker scheduling delay.

  This is basically to make sure that setting environment variables
  `TF_COORDINATOR_SCHEDULE_START_DELAY` and
  `TF_COORDINATOR_SCHEDULE_START_DELAY_MAX` will cause any failure.
  """

  @classmethod
  def setUpClass(cls):
    super(ScheduleStartDelayTest, cls).setUpClass()
    os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY'] = '2'
    os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY_MAX'] = '4'
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)
    cls.strategy = cls.coordinator.strategy

  @classmethod
  def tearDownClass(cls):
    del os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY']
    del os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY_MAX']
    super(ScheduleStartDelayTest, cls).tearDownClass()


class ErrorReportingTest(TestCaseWithErrorReportingThread):

  @classmethod
  def setUpClass(cls):
    super(ErrorReportingTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)
    cls.strategy = cls.coordinator.strategy

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
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

  def testScheduleRaiseError(self):
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      while True:
        self.coordinator.schedule(self._normal_function)

  def testScheduleRaiseErrorWithMultipleFailure(self):
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      while True:
        self.coordinator.schedule(self._error_function)
    self.coordinator.join()

  def testErrorWillbeCleared(self):
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

  def testRemoteValueReturnError(self):
    self.skipTest('TODO(b/211502459): Fix this in OSS test.')

    result = self.coordinator.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      result.fetch()

    # Clear the error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

  def testInputError(self):

    worker_local_val = self.coordinator._create_per_worker_resources(
        self._error_function)

    @def_function.function
    def func(x):
      return x + 1

    result = self.coordinator.schedule(func, args=(worker_local_val,))
    with self.assertRaises(coordinator_lib.ClosureInputError):
      self.coordinator.join()

    with self.assertRaises(coordinator_lib.ClosureInputError):
      result.fetch()

  def testErroredInputNotUsed(self):
    input_0 = self.coordinator._create_per_worker_resources(
        self._normal_function)

    self.coordinator._create_per_worker_resources(
        self._error_function)

    @def_function.function
    def func(x):
      return x + 1

    result = self.coordinator.schedule(func, args=(input_0,))

    # It should not raise.
    self.coordinator.join()
    result.fetch()

  def testCancellation(self):
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    long_function = self.coordinator.schedule(self._long_function)
    self.coordinator.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

    with self.assertRaises(errors.CancelledError):
      long_function.fetch()

    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.join()

  def testResourceCanStillbeUsedAfterCancellation(self):

    def input_fn():
      return dataset_ops.DatasetV2.range(0, 5)

    per_worker_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      return next(iterator)

    self.coordinator.schedule(worker_fn, args=(per_worker_iterator,))
    self.coordinator.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

    self.coordinator.schedule(worker_fn, args=(per_worker_iterator,))
    self.coordinator.join()


class LimitedClosureQueueErrorTest(ErrorReportingTest):
  """Test error reporting works with explicit maximum closure queue size.

  Execute the same set of test cases as in ErrorReportingTest, with an explicit
  size limit for the closure queue.
  """

  @classmethod
  def setUpClass(cls):
    super(LimitedClosureQueueErrorTest, cls).setUpClass()
    coordinator_lib._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)
    cls.strategy = cls.coordinator.strategy

    with cls.coordinator.strategy.scope():
      cls.iteration = variables.Variable(initial_value=0.0)


class StrategyIntegrationTest(test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(StrategyIntegrationTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=1, num_ps=1)
    cls.strategy = cls.coordinator.strategy

  def testRunNotUsedWithClusterCoordinatorSchedule(self):

    @def_function.function
    def input_fn():
      return dataset_ops.DatasetV2.range(1, 3)

    with self.strategy.scope():
      v = variables.Variable(initial_value=1, dtype=dtypes.int64)

      def replica_fn(input_tensor):
        return input_tensor + v, input_tensor - v

      @def_function.function
      def worker_fn(iterator):
        return self.strategy.run(replica_fn, args=(next(iterator),))

    per_worker_dataset = self.coordinator.create_per_worker_dataset(input_fn)

    @contextlib.contextmanager
    def _assert_logs_usage_warning():
      with self.assertLogs(level='WARNING') as logs:
        yield

      self.assertIn(
          'A `tf.distribute.experimental.ParameterServerStrategy` method is '
          'invoked without using `ClusterCoordinator.schedule`. If you are not '
          'tracing a tf.function, this method is possibly executed on the '
          'coordinator, which can be slow. To properly dispatch functions to '
          'run on workers, methods like `run` or `reduce` should be used '
          'within a function passed to `tf.distribute.experimental.coordinator.'
          'ClusterCoordinator.schedule`.', logs.output[0])

    with _assert_logs_usage_warning():
      # Invoking `run` without `coordinator.schedule` should result in a
      # warning.
      self.strategy.run(
          replica_fn, args=(constant_op.constant(1, dtype=dtypes.int64),))

    # A proper `schedule` should succeed.
    rv = self.coordinator.schedule(worker_fn, args=(iter(per_worker_dataset),))

    with _assert_logs_usage_warning():
      # Invoking `run` without `coordinator.schedule` again should result in a
      # warning.
      self.strategy.run(
          replica_fn, args=(constant_op.constant(1, dtype=dtypes.int64),))

    all_results = [(2, 0)] * self.strategy.num_replicas_in_sync
    expected_result = []
    for i in range(self.strategy.num_replicas_in_sync):
      expected_result.append(all_results[i])

    self.assertAllEqual(
        tuple(expected_result),
        self.strategy.experimental_local_results(rv.fetch()))

  def testBasicVariableAssignment(self):
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

    results = self.coordinator.schedule(worker_fn)
    logging.info('Results of experimental_run_v2: %f',
                 self.coordinator.fetch(results))

    self.assertAlmostEqual(v1.read_value().numpy(), 0.1, delta=1e-6)
    self.assertAlmostEqual(v2.read_value().numpy(), 0.8, delta=1e-6)

  def testRunAndReduce(self):
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(initial_value=1.)

      expected_result = (4. * self.strategy.num_replicas_in_sync,
                         2. * self.strategy.num_replicas_in_sync)

      @def_function.function
      def worker_fn(input_tensor):

        def replica_fn(input_tensor):
          # Within `replica_fn`, it has to be in a replica context.
          self.assertFalse(
              distribute_lib.in_cross_replica_context())
          return input_tensor + v, input_tensor - v

        run_result = self.strategy.run(replica_fn, args=(input_tensor,))
        reduced_result = self.strategy.reduce('SUM', run_result, axis=None)
        check_ops.assert_equal_v2(reduced_result, expected_result)
        return reduced_result

      # Asserting scheduling in scope has the expected behavior.
      result = self.coordinator.schedule(
          worker_fn, args=(constant_op.constant(3.),))
      self.assertIsInstance(result, coordinator_lib.RemoteValue)
      self.assertEqual(result.fetch(), expected_result)

    # Asserting scheduling out of scope has the expected behavior.
    result = self.coordinator.schedule(
        worker_fn, args=(constant_op.constant(3.),))
    self.assertEqual(result.fetch(), expected_result)

  def testRunAndReduceWithAssignAdd(self):
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(initial_value=1.)
      v1 = variables.Variable(
          initial_value=0.,
          aggregation=variable_scope.VariableAggregation.ONLY_FIRST_REPLICA)

      expected_result = (4. * self.strategy.num_replicas_in_sync,
                         2. * self.strategy.num_replicas_in_sync)

      @def_function.function
      def worker_fn(input_tensor):

        def replica_fn(input_tensor):
          # Within `replica_fn`, it has to be in a replica context.
          self.assertFalse(
              distribute_lib.in_cross_replica_context())

          v1.assign_add(input_tensor)
          return input_tensor + v, input_tensor - v

        run_result = self.strategy.run(replica_fn, args=(input_tensor,))
        reduced_result = self.strategy.reduce('SUM', run_result, axis=None)
        check_ops.assert_equal_v2(reduced_result, expected_result)
        return reduced_result

      # Asserting scheduling in scope has the expected behavior.
      result = self.coordinator.schedule(
          worker_fn, args=(constant_op.constant(3.),))
      self.assertIsInstance(result, coordinator_lib.RemoteValue)
      self.assertEqual(result.fetch(), expected_result)

    # Asserting scheduling out of scope has the expected behavior.
    result = self.coordinator.schedule(
        worker_fn, args=(constant_op.constant(3.),))
    self.assertEqual(result.fetch(), expected_result)
    self.assertEqual(v1, 6.)

  def testVariableAggregation(self):
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(
          initial_value=1.,
          aggregation=variable_scope.VariableAggregation.SUM)

      @def_function.function
      def worker_fn():

        def replica_fn():
          value = math_ops.cast(
              distribute_lib.get_replica_context()
              .replica_id_in_sync_group + 1, v.dtype)
          v.assign(value)

        self.strategy.run(replica_fn)

      self.coordinator.schedule(worker_fn)
      self.coordinator.join()
      expected_result = 0.
      for i in range(self.strategy.num_replicas_in_sync):
        expected_result = expected_result + i + 1
      self.assertEqual(v, expected_result)

  def testVariableCaching(self):
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(
          initial_value=1.,
          aggregation=variable_scope.VariableAggregation.ONLY_FIRST_REPLICA)

      # Test read value inside caching scope
      with distribute_utils.cache_variable_reads():
        v.read_value()  # Reads value 1.0
        v.assign(constant_op.constant(5.0))  # v changes to 5.0
        self.assertEqual(v.read_value(), 1.0)  # should be cached 1.0 value.

      # Reset v to 2.0
      v.assign(2.0)

      # Test convert to tensor value inside caching scope
      with distribute_utils.cache_variable_reads():
        t = v * 3.0
        self.assertEqual(t, 6.0)
        v.assign(3.0)
        t1 = v * 3.0
        self.assertEqual(t1, 6.0)  # should be cached 2.0 * 3.0 value.

      # Reset v to 1.0
      v.assign(1.0)

      # Verify caching scope inside tf.function
      @def_function.function
      def worker_fn():
        with distribute_utils.cache_variable_reads():
          def replica_fn():
            t = v.read_value()  # Reads value 1.0
            v.assign(constant_op.constant(5.0))  # v changes to 5.0
            t = v.read_value()  # should return 1.0
            return t  # Should be 1.0 instead of 5.0

          return self.strategy.run(replica_fn)

      result = self.coordinator.schedule(worker_fn)
      result = result.fetch()
      expected_result = 1.
      self.assertEqual(result, expected_result)

      # Verify that v.read_value works as expected outside of scope.
      v.assign(4.0)
      self.assertEqual(v.read_value(), 4.0)

      v.assign(constant_op.constant(2.0))  # v changes to 2.0
      # Check with scope outside of tf function and check that cache is reset
      @def_function.function
      def worker_fn1():
        def replica_fn():
          t = v.read_value()  # Reads value 2.0 ==> Should be cached
          v.assign(constant_op.constant(5.0))  # v changes to 5.0
          t = v.read_value()  # should return cached value 2.0
          return t  # Should be 2.0 instead of 5.0

        return self.strategy.run(replica_fn)

      with distribute_utils.cache_variable_reads():
        result = self.coordinator.schedule(worker_fn1)
      result = result.fetch()
      expected_result = 2.
      self.assertEqual(result, expected_result)

    # Verify scope nesting is not permitted.
    with self.assertRaises(ValueError):
      with distribute_utils.cache_variable_reads():
        with distribute_utils.cache_variable_reads():
          v.read_value()

  @parameterized.parameters(True, False)
  def testDistributedDatasetInsidePerWorkerDatasetFn(self, from_function):
    if from_function:

      def per_worker_dataset_fn():
        dataset_fn = lambda _: dataset_ops.DatasetV2.range(1, 11).batch(4)
        return self.strategy.distribute_datasets_from_function(dataset_fn)
    else:

      def per_worker_dataset_fn():
        dataset = dataset_ops.DatasetV2.range(1, 11).batch(4)
        return self.strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def worker_fn(iterator):
      return self.strategy.experimental_local_results(next(iterator))

    per_worker_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(per_worker_dataset),))
    result = result.fetch()
    expected_result = array_ops.split(
        math_ops.range(1., 5.),
        num_or_size_splits=self.strategy.num_replicas_in_sync,
        axis=0)

    self.assertAllEqual(result, (expected_result))

  @parameterized.parameters(True, False)
  def testPassDistributedDatasetToCreatePerWorkerDataset(self, from_function):
    if from_function:
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(1, 11).batch(4)
      distributed_dataset = self.strategy.distribute_datasets_from_function(
          dataset_fn)
    else:
      dataset = dataset_ops.DatasetV2.range(1, 11).batch(4)
      distributed_dataset = self.strategy.experimental_distribute_dataset(
          dataset)

    @def_function.function
    def worker_fn(iterator):
      return self.strategy.experimental_local_results(next(iterator))

    per_worker_dataset = self.coordinator.create_per_worker_dataset(
        distributed_dataset)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(per_worker_dataset),))
    result = result.fetch()
    expected_result = array_ops.split(
        math_ops.range(1., 5.),
        num_or_size_splits=self.strategy.num_replicas_in_sync,
        axis=0)

    self.assertAllEqual(result, (expected_result))

  def testDistributeDatasetsFromFunction(self):

    def per_worker_dataset_fn():

      def input_worker_device_fn(input_context):
        self.assertIsNotNone(input_context)
        return dataset_ops.DatasetV2.range(1, 11).batch(1)

      return self.strategy.distribute_datasets_from_function(
          input_worker_device_fn)

    @def_function.function
    def worker_fn(iterator):
      result = self.strategy.experimental_local_results(next(iterator))
      return result

    distributed_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    result = result.fetch()
    expected_result = []
    for i in range(self.strategy.num_replicas_in_sync):
      expected_result.append([1 + i])
    self.assertAllEqual(result, expected_result)

  def testAsyncScheduleWithDistributedDataset(self):

    def input_fn():
      dataset = dataset_ops.DatasetV2.from_tensor_slices([2.]).repeat().batch(
          self.strategy.num_replicas_in_sync)
      return self.strategy.experimental_distribute_dataset(dataset)

    with self.strategy.scope():
      v = variables.Variable(initial_value=[0], dtype=dtypes.float32)

    # TODO(yuefengz): the following tf.function has a return value which is None
    # in its structured_outputs.
    @def_function.function
    def worker_fn(iterator):
      x = next(iterator)
      # Reduce to convert PerReplica values to single value
      reduced_value = self.strategy.reduce('MEAN', x, axis=None)
      v.assign_add(reduced_value)

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)

    iterator = iter(distributed_dataset)

    # Verifying joining without any scheduling doesn't hang.
    self.coordinator.join()
    self.assertAllEqual(v.read_value(), (0,))

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))
    self.coordinator.join()

    # With 5 addition it should be 2*5 = 10.
    self.assertAllEqual(
        self.strategy.experimental_local_results(v.read_value()), ([[10]]))

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))

    # Verifying multiple join is fine.
    self.coordinator.join()
    self.coordinator.join()
    self.coordinator.join()

    self.assertTrue(self.coordinator.done())

    # Likewise, it's now 20.
    self.assertAllEqual(
        self.strategy.experimental_local_results(v.read_value()), ([[20]]))

  def testInputFunctionWithMapWithDistributedDataset(self):
    self._map_fn_tracing_count = 0

    def input_fn():

      def map_fn(x):
        self._map_fn_tracing_count += 1
        return x + 10

      dataset = dataset_ops.DatasetV2.range(0, 10).batch(
          self.strategy.num_replicas_in_sync).map(map_fn)
      return self.strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def worker_fn(iterator):
      return next(iterator)

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))

    expected_result = array_ops.split(
        math_ops.range(10., 10. + self.strategy.num_replicas_in_sync),
        num_or_size_splits=self.strategy.num_replicas_in_sync,
        axis=0)

    self.assertAllEqual(
        self.strategy.experimental_local_results(result.fetch()),
        tuple(expected_result))
    self.assertEqual(self._map_fn_tracing_count, 1)

  def testPerWorkerDistributeDatasetsElementSpec(self):

    def per_worker_dataset_fn():
      return self.strategy.distribute_datasets_from_function(
          lambda _: dataset_ops.DatasetV2.from_tensor_slices([1, 2]))

    dataset = dataset_ops.DatasetV2.from_tensor_slices([1, 2])
    per_worker_distribute_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)

    self.assertAllEqual(
        # Converts to PerReplicaSpec when num_replicas_in_sync are > 1
        input_lib._create_distributed_tensor_spec(self.strategy,
                                                  dataset.element_spec),
        per_worker_distribute_dataset.element_spec)

  def testPerWorkerDistributedIteratorTypeSpec(self):
    self._tracing_count = 0

    def per_worker_dataset_fn():
      self._tracing_count += 1
      return self.strategy.distribute_datasets_from_function(
          lambda _: dataset_ops.DatasetV2.range(1, 2))

    @def_function.function
    def worker_fn(iterator):
      return next(iterator)

    distributed_iterator = iter(
        self.coordinator.create_per_worker_dataset(per_worker_dataset_fn))
    worker_fn.get_concrete_function(distributed_iterator)

    self.coordinator.schedule(worker_fn, args=(distributed_iterator,))
    self.assertEqual(self._tracing_count, 1)

  def testPerWorkerDatasetBuild(self):
    # Test that we can use Dataset type as input to worker_fn.
    @def_function.function
    def worker_fn(dataset):
      return next(iter(dataset))

    dataset_vals = [1, 2]
    dataset = dataset_ops.DatasetV2.from_tensor_slices(dataset_vals)
    per_worker_dataset = self.coordinator.create_per_worker_dataset(dataset)
    per_worker_dataset = per_worker_dataset.build()

    result = self.coordinator.schedule(worker_fn, args=(per_worker_dataset,))
    self.coordinator.join()
    result = result.fetch()
    self.assertEqual(result, dataset_vals[0])

    # Test that the build() output type specs match the input Dataset spec.
    for value in per_worker_dataset._values:
      self.assertEqual(value._type_spec, dataset._type_spec)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
