# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Memory leak detection utility."""

from tensorflow.python.framework.python_memory_checker import _PythonMemoryChecker
from tensorflow.python.profiler import trace
from tensorflow.python.util import tf_inspect

try:
  from tensorflow.python.platform.cpp_memory_checker import _CppMemoryChecker as CppMemoryChecker  # pylint:disable=g-import-not-at-top
except ImportError:
  CppMemoryChecker = None


def _get_test_name_best_effort():
  """If available, return the current test name. Otherwise, `None`."""
  for stack in tf_inspect.stack():
    function_name = stack[3]
    if function_name.startswith('test'):
      try:
        class_name = stack[0].f_locals['self'].__class__.__name__
        return class_name + '.' + function_name
      except:  # pylint:disable=bare-except
        pass

  return None


# TODO(kkb): Also create decorator versions for convenience.
class MemoryChecker(object):
  """Memory leak detection class.

  This is a utility class to detect Python and C++ memory leaks. It's intended
  for both testing and debugging. Basic usage:

  >>> # MemoryChecker() context manager tracks memory status inside its scope.
  >>> with MemoryChecker() as memory_checker:
  >>>   tensors = []
  >>>   for _ in range(10):
  >>>     # Simulating `tf.constant(1)` object leak every iteration.
  >>>     tensors.append(tf.constant(1))
  >>>
  >>>     # Take a memory snapshot for later analysis.
  >>>     memory_checker.record_snapshot()
  >>>
  >>> # `report()` generates a html graph file showing allocations over
  >>> # snapshots per every stack trace.
  >>> memory_checker.report()
  >>>
  >>> # This assertion will detect `tf.constant(1)` object leak.
  >>> memory_checker.assert_no_leak_if_all_possibly_except_one()

  `record_snapshot()` must be called once every iteration at the same location.
  This is because the detection algorithm relies on the assumption that if there
  is a leak, it's happening similarly on every snapshot.
  """

  @trace.trace_wrapper
  def __enter__(self):
    self._python_memory_checker = _PythonMemoryChecker()
    if CppMemoryChecker:
      self._cpp_memory_checker = CppMemoryChecker(_get_test_name_best_effort())
    return self

  @trace.trace_wrapper
  def __exit__(self, exc_type, exc_value, traceback):
    if CppMemoryChecker:
      self._cpp_memory_checker.stop()

  # We do not enable trace_wrapper on this function to avoid contaminating
  # the snapshot.
  def record_snapshot(self):
    """Take a memory snapshot for later analysis.

    `record_snapshot()` must be called once every iteration at the same
    location. This is because the detection algorithm relies on the assumption
    that if there is a leak, it's happening similarly on every snapshot.

    The recommended number of `record_snapshot()` call depends on the testing
    code complexity and the allcoation pattern.
    """
    self._python_memory_checker.record_snapshot()
    if CppMemoryChecker:
      self._cpp_memory_checker.record_snapshot()

  @trace.trace_wrapper
  def report(self):
    """Generates a html graph file showing allocations over snapshots.

    It create a temporary directory and put all the output files there.
    If this is running under Google internal testing infra, it will use the
    directory provided the infra instead.
    """
    self._python_memory_checker.report()
    if CppMemoryChecker:
      self._cpp_memory_checker.report()

  @trace.trace_wrapper
  def assert_no_leak_if_all_possibly_except_one(self):
    """Raises an exception if a leak is detected.

    This algorithm classifies a series of allocations as a leak if it's the same
    type(Python) or it happens at the same stack trace(C++) at every snapshot,
    but possibly except one snapshot.
    """

    self._python_memory_checker.assert_no_leak_if_all_possibly_except_one()
    if CppMemoryChecker:
      self._cpp_memory_checker.assert_no_leak_if_all_possibly_except_one()

  @trace.trace_wrapper
  def assert_no_new_python_objects(self, threshold=None):
    """Raises an exception if there are new Python objects created.

    It computes the number of new Python objects per type using the first and
    the last snapshots.

    Args:
      threshold: A dictionary of [Type name string], [count] pair. It won't
        raise an exception if the new Python objects are under this threshold.
    """
    self._python_memory_checker.assert_no_new_objects(threshold=threshold)
