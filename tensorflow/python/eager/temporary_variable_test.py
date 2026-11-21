# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests execution-local lifetimes of temporary variables in eager functions."""

import threading

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class TemporaryVariableTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('sync', context.SYNC), ('async', context.ASYNC))
  def testDatasetExhaustionDoesNotPoisonNextInvocation(self, mode):
    name = self.id()

    @polymorphic_function.function(autograph=False)
    def consume(iterator):
      temporary = gen_state_ops.temporary_variable([], dtypes.int64, name)
      # Force the EOF error to occur after allocation, without sleeps or a
      # memory-pressure heuristic to trigger the Grappler rewrite.
      with ops.control_dependencies([temporary]):
        value = next(iterator)
      assigned = gen_state_ops.assign(temporary, value)
      with ops.control_dependencies([assigned]):
        return gen_state_ops.destroy_temporary_variable(temporary, name)

    options = options_lib.Options()
    options.threading.private_threadpool_size = 1
    dataset = dataset_ops.Dataset.range(2).with_options(options)
    with context.execution_mode(mode):
      for _ in range(3):
        iterator = iter(dataset)
        self.assertEqual(0, self.evaluate(consume(iterator)))
        self.assertEqual(1, self.evaluate(consume(iterator)))
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(consume(iterator))
        context.context().executor.clear_error()

  @parameterized.named_parameters(
      ('sync', context.SYNC), ('async', context.ASYNC))
  def testErrorDoesNotPoisonNextInvocation(self, mode):
    name = self.id()

    @polymorphic_function.function(autograph=False)
    def checked_value(value):
      temporary = gen_state_ops.temporary_variable([], dtypes.int32, name)
      with ops.control_dependencies([temporary]):
        checked = check_ops.assert_positive(value)
      with ops.control_dependencies([checked]):
        assigned = gen_state_ops.assign(temporary, value)
      with ops.control_dependencies([assigned]):
        return gen_state_ops.destroy_temporary_variable(temporary, name)

    with context.execution_mode(mode):
      for value in (3, -1, 4, -1, 5):
        value_tensor = constant_op.constant(value)
        if value > 0:
          self.assertEqual(value, self.evaluate(checked_value(value_tensor)))
        else:
          with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(checked_value(value_tensor))
          context.context().executor.clear_error()

  def testConcurrentInvocationsUseIndependentTemporaryVariables(self):
    name = self.id()
    barrier = threading.Barrier(2, timeout=10)

    def wait_for_both_invocations():
      barrier.wait()
      return 0

    @polymorphic_function.function(autograph=False)
    def concurrent_value(value):
      temporary = gen_state_ops.temporary_variable([], dtypes.int32, name)
      assigned = gen_state_ops.assign(temporary, value)
      with ops.control_dependencies([assigned]):
        ready = script_ops.eager_py_func(
            wait_for_both_invocations, [], dtypes.int32)
      with ops.control_dependencies([ready]):
        return gen_state_ops.destroy_temporary_variable(temporary, name)

    concrete = concurrent_value.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.int32))
    values = [constant_op.constant(11), constant_op.constant(29)]
    results = [None, None]

    def run(index):
      results[index] = self.evaluate(concrete(values[index]))

    threads = [self.checkedThread(run, args=(i,)) for i in range(2)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    self.assertAllEqual([11, 29], results)

  def testCancellationDoesNotPoisonNextInvocation(self):
    name = self.id()
    allocated = threading.Event()
    queue = data_flow_ops.FIFOQueue(1, dtypes.int32)

    def notify_allocated():
      allocated.set()
      return 0

    @polymorphic_function.function(autograph=False)
    def consume():
      temporary = gen_state_ops.temporary_variable([], dtypes.int32, name)
      with ops.control_dependencies([temporary]):
        ready = script_ops.eager_py_func(notify_allocated, [], dtypes.int32)
      with ops.control_dependencies([ready]):
        value = queue.dequeue()
      assigned = gen_state_ops.assign(temporary, value)
      with ops.control_dependencies([assigned]):
        return gen_state_ops.destroy_temporary_variable(temporary, name)

    manager = cancellation.CancellationManager()
    cancelable = manager.get_cancelable_function(
        consume.get_concrete_function())

    def cancel_after_allocation():
      # Cancel even if the notification times out, so a failing test cannot
      # leave the dequeue blocked indefinitely.
      ready = allocated.wait(timeout=10)
      manager.start_cancel()
      self.assertTrue(ready)

    thread = self.checkedThread(cancel_after_allocation)
    thread.start()
    with self.assertRaises(errors.CancelledError):
      self.evaluate(cancelable())
    thread.join()
    queue.enqueue(7)
    self.assertEqual(7, self.evaluate(consume()))

  def testTensorArrayCanOutliveFunctionInvocation(self):
    name = self.id()

    @polymorphic_function.function(autograph=False)
    def create():
      temporary = gen_state_ops.temporary_variable([], dtypes.float32, name)
      with ops.control_dependencies([temporary]):
        handle, flow = gen_data_flow_ops.tensor_array_v3(
            size=1, dtype=dtypes.float32)
      flow = gen_data_flow_ops.tensor_array_write_v3(handle, 0, 3.0, flow)
      return handle, flow

    @polymorphic_function.function(
        autograph=False,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.resource),
                         tensor_spec.TensorSpec(None, dtypes.float32)])
    def read_and_close(handle, flow):
      value = gen_data_flow_ops.tensor_array_read_v3(
          handle, 0, flow, dtypes.float32)
      with ops.control_dependencies([value]):
        close = gen_data_flow_ops.tensor_array_close_v3(handle)
      with ops.control_dependencies([close]):
        return array_ops.identity(value)

    self.assertEqual(3.0, self.evaluate(read_and_close(*create())))

  def testStackCanOutliveFunctionInvocation(self):
    name = self.id()

    @polymorphic_function.function(autograph=False)
    def create():
      temporary = gen_state_ops.temporary_variable([], dtypes.float32, name)
      with ops.control_dependencies([temporary]):
        handle = gen_data_flow_ops.stack_v2(1, dtypes.float32)
      value = gen_data_flow_ops.stack_push_v2(handle, 3.0)
      with ops.control_dependencies([value]):
        return array_ops.identity(handle)

    @polymorphic_function.function(
        autograph=False,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.resource)])
    def pop_and_close(handle):
      value = gen_data_flow_ops.stack_pop_v2(handle, dtypes.float32)
      with ops.control_dependencies([value]):
        close = gen_data_flow_ops.stack_close_v2(handle)
      with ops.control_dependencies([close]):
        return array_ops.identity(value)

    self.assertEqual(3.0, self.evaluate(pop_and_close(create())))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
