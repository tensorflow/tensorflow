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
"""Multi-process runner for testing purpose."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time

import six
from six.moves import queue as Queue

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.platform import test
from tensorflow.python.util import nest

# _ProcessStatusInfo contains process status information. When is_successful
# attribute is True, the subprocess has ended successfully, or if False, the
# exception stack trace info is stored in exc_info to pass on to parent process
# to be re-raised.
_ProcessStatusInfo = collections.namedtuple(
    '_ProcessStatusInfo', ['task_type', 'is_successful', 'exc_info'])

# Process status queue is used by `multi_process_runner` internally for
# communication from subprocesses to the parent process.
PROCESS_STATUS_QUEUE = 'process_status_queue'

# Return value queue is intended to be used by users of `multi_process_runner`
# for the process function to return information to the caller of
# `multi_process_runner.run()`.
RETURN_VALUE_QUEUE = 'return_value_queue'

# Standard stream queue is used by `multi_process_runner` to collect
# information streamed to stdout and stderr to be reported back to the
# parent process.
STD_STREAM_QUEUE = 'std_stream_queue'

# Inter-process queue is used for communications between subprocesses.
INTER_PROCESS_QUEUE = 'inter_process_queue'

# Parent-to-sub queue is used for communications from parent to subprocess.
# Currently this is only used to terminate subprocesses.
PARENT_TO_SUB_QUEUE = 'parent_to_sub_queue'


class _LogCollector(object):
  """Tool to collect logs before sending them to std stream."""

  def __init__(self, original_stream):
    self.log = []
    self.original_stream = original_stream

  def write(self, data):
    self.log.append(data)
    self.original_stream.write(data)

  def flush(self, *args, **kwargs):
    self.original_stream.flush(*args, **kwargs)


class MultiProcessRunner(object):
  """A utility class to start multiple processes to simulate a cluster.

  We need to use multiple processes to simulate a cluster in TF 2.0 tests
  because TF 2.0 has some process-global data structures that have to be
  separated by processes. We also need child processes to test out our fault
  tolerance because shutting down a standard TensorFlow server within its
  process is not supported.

  Note: the main test program that uses this runner class must run main program
  via `test_main` defined in this file. Using this runner in non-test binaries
  is not supported yet.

  This class is not thread-safe. Child processes will inherit TF2 behavior flag.
  """

  def __init__(self,
               proc_func,
               cluster_spec,
               max_run_time=None,
               capture_std_stream=False,
               grpc_fail_fast=False,
               args=None,
               kwargs=None):
    """Creates a multi-process runner.

    Args:
      proc_func: Function to be run on child processes. This will be run on
        processes for all task types.
      cluster_spec: Dict for cluster spec. The following is an example of
        cluster with three workers and two ps's.
        {"worker": ["worker0.example.com:2222",
                    "worker1.example.com:2222",
                    "worker2.example.com:2222"],
         "ps": ["ps0.example.com:2222",
                "ps1.example.com:2222"]}
      max_run_time: If set, child processes is forced to exit at approximately
        this many seconds after `start` is called. We achieve this through
        `signal.alarm()` api. Note that this is best effort at Python level
        since Python signal handler does not get executed when it runs lower
        level C/C++ code. So it can be delayed for arbitrarily long time.
      capture_std_stream: Boolean, whether the messages streamed to stdout and
        stderr in subprocesses are captured.
      grpc_fail_fast: Whether GRPC connection between processes should fail
        without retrying. Defaults to False.
      args: Positional arguments to be sent to functions run on processes.
      kwargs: Keyword arguments to be sent to functions run on processes.

    Raises:
      RuntimeError: if `multi_process_runner.test_main()` is not called.
      ValueError: if there are more than one chief in the `cluster_spec`.
    """
    assert cluster_spec is not None
    if 'chief' in cluster_spec and len(cluster_spec['chief']) > 1:
      raise ValueError('If chief exists in the cluster, there must be at most '
                       'one chief. Current `cluster_spec` has {} chiefs.'
                       .format(len(cluster_spec['chief'])))

    assert callable(proc_func)

    if not multi_process_lib.using_context_manager():
      raise RuntimeError('`multi_process_runner` is not initialized. '
                         'Please call `multi_process_runner.test_main()` '
                         'within `if __name__ == \'__main__\':` block '
                         'in your python module to properly initialize '
                         '`multi_process_runner`.')

    self._proc_func = proc_func
    self._cluster_spec = cluster_spec
    self._max_run_time = max_run_time
    self._capture_std_stream = capture_std_stream
    self._grpc_fail_fast = grpc_fail_fast
    self._args = args or ()
    self._kwargs = kwargs or {}
    self._outstanding_subprocess_count = 0

    # Child processes should have the same v2 and eager behavior.
    self._v2_enabled = tf2.enabled()
    self._executing_eagerly = context.executing_eagerly()

  @contextlib.contextmanager
  def _runtime_mode(self):
    if self._executing_eagerly:
      with context.eager_mode():
        yield
    else:
      with context.graph_mode():
        yield

  def _finish_process(self, process_status_info, return_value, stdout_collector,
                      stderr_collector):
    """Adds data to queues before program exits."""
    # Clear the alarm.
    signal.alarm(0)

    # When chief exists in the cluster, there must only be one chief and it
    # needs to reach this point before any other exits. The reason is chief
    # would continue to ping ps/workers if ps/workers exit before chief does,
    # and this results in connection error flakiness.
    # TODO(rchao): Modify this mechanism so that parent sends out the signal
    # to terminate the subprocesses to have better control over the cases where
    # fault tolerance is being tested. After the start of such signal from the
    # parent, the errors should be ignored.
    if 'chief' in self._cluster_spec:
      if process_status_info.task_type == 'chief':
        # When executed by chief, for each task in the cluster, except for
        # chief, add an item in the queue as a notification for those tasks to
        # know they can continue to terminate the process.
        for _ in range(len(nest.flatten(self._cluster_spec)) - 1):
          self._get_inter_process_queue().put(True)
      else:
        # When executed by non-chief, they need to block until the signal from
        # chief is received.
        self._get_inter_process_queue().get()

    if return_value is not None:
      self._add_return_data(return_value)
    if self._capture_std_stream:
      # If stdout and stderr are to be collected, add them to std stream
      # queue.
      self._add_std_stream_data_flattened(stdout_collector.log)
      self._add_std_stream_data_flattened(stderr_collector.log)
    self._get_process_status_queue().put(process_status_info)

  def _message_checking_func(self, task_type, task_id, stdout_collector,
                             stderr_collector):
    """A function that regularly checks messages from parent process."""
    while True:
      try:
        message = self._get_parent_to_sub_queue().get(block=False)
        # Currently the only possible message is termination.
        assert message.startswith('terminate')
        if message == 'terminate {} {}'.format(task_type, task_id):
          break
        else:
          # If the message is not targeting this process, put it back to the
          # queue.
          self._get_parent_to_sub_queue().put(message)
          time.sleep(1)
      except Queue.Empty:
        time.sleep(0.1)
    self._finish_process(
        _ProcessStatusInfo(
            task_type=task_type, is_successful=True, exc_info=None), None,
        stdout_collector, stderr_collector)
    # `os._exit(0)` is used to more reliably terminate a subprocess.
    os._exit(0)  # pylint: disable=protected-access

  def _proc_func_wrapper(self, proc_func, task_type, task_id,
                         per_process_cluster_spec, *arg, **kwargs):
    """The wrapper function that actually gets run in child process(es)."""

    if self._capture_std_stream:
      # TODO(yuefengz): consider a lighter way of capturing std streams.
      stdout_collector = _LogCollector(sys.__stdout__)
      stderr_collector = _LogCollector(sys.__stderr__)
      sys.stdout = stdout_collector
      sys.stderr = stderr_collector
    else:
      stdout_collector = None
      stderr_collector = None

    # The thread will be dedicated to checking messages from parent process.
    threading.Thread(
        target=self._message_checking_func,
        args=(task_type, task_id, stdout_collector, stderr_collector)).start()

    os.environ['GRPC_FAIL_FAST'] = str(self._grpc_fail_fast)
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': per_process_cluster_spec,
        'task': {
            'type': task_type,
            'index': task_id,
        }
    })

    if self._v2_enabled:
      v2_compat.enable_v2_behavior()

    return_value = None

    if self._max_run_time is not None:
      # Register an sigalarm handler to exit the process when it reaches
      # `timeout` seconds. A program reaching `timeout` doesn't necessarily
      # indicate an issue.
      def handler(signum, frame):
        del signum, frame
        self._finish_process(
            _ProcessStatusInfo(
                task_type=task_type, is_successful=True, exc_info=None), None,
            stdout_collector, stderr_collector)
        # `os._exit(0)` is used to more reliably terminate a subprocess.
        os._exit(0)  # pylint: disable=protected-access

      signal.signal(signal.SIGALRM, handler)
      signal.alarm(self._max_run_time)

    try:
      with self._runtime_mode():
        return_value = proc_func(*arg, **kwargs)
    except Exception:  # pylint: disable=broad-except
      # Capture all exceptions to be reported to parent process.
      self._finish_process(
          _ProcessStatusInfo(
              task_type=task_type, is_successful=False,
              exc_info=sys.exc_info()), return_value, stdout_collector,
          stderr_collector)

      # Re-raise the exception in addition to reporting it to the parent
      # process, so that even if `--test_timeout` flag is set and the
      # error doesn't make it to be shown in parent process before bazel's
      # timeout, the log would still show what happens in this subprocess,
      # instead of silently suppressing the error due to early bazel
      # timeout. Raising an error in the subprocess produces stack trace in
      # the log, but the program continues running.
      raise

    self._finish_process(
        _ProcessStatusInfo(
            task_type=task_type, is_successful=True, exc_info=None),
        return_value, stdout_collector, stderr_collector)

  def start(self):
    """Starts processes, one for each task in `cluster_spec`.

    If 'chief' job exists in the cluster, it is guaranteed that 'chief'
    process exits before other jobs to prevent chief from continuing to connect
    to them which causes error.
    """
    for task_type, addresses in self._cluster_spec.items():
      for task_id, _ in enumerate(addresses):
        p = multi_process_lib.Process(
            target=self._proc_func_wrapper,
            args=(self._proc_func, task_type, task_id, self._cluster_spec) +
            self._args,
            kwargs=self._kwargs)
        p.start()
        self._outstanding_subprocess_count += 1

  def start_single_process(self,
                           task_type,
                           task_id,
                           proc_func=None,
                           updated_cluster_spec=None,
                           args=None,
                           kwargs=None):
    """Starts a single process.

    This starts a process in the cluster with the task type, task id, and the
    process function (`proc_func`). If process function is `None`, the function
    provided at `__init__` will be used. If `updated_cluster_spec` is not
    `None`, the cluster spec used by this subprocess will be updated.

    TODO(rchao): It is meant that all subprocesses will be updated with the new
    cluster spec, but this has yet to be implemented. At this time only the
    newly started subprocess picks up this updated cluster spec.

    Args:
      task_type: The task type.
      task_id: The task id.
      proc_func: The process function to be run on the newly started
        process. If `None`, the function provided at `__init__` will be used.
      updated_cluster_spec: If not `None`, the cluster spec used by this
        subprocess will be updated.
      args: Optional positional arguments to be supplied in `proc_func`.
      kwargs: Optional keyword arguments to be supplied in `proc_func`.
    """
    self._cluster_spec = updated_cluster_spec or self._cluster_spec
    proc_func = proc_func or self._proc_func
    p = multi_process_lib.Process(
        target=self._proc_func_wrapper,
        args=(proc_func, task_type, task_id, self._cluster_spec) + (args or ()),
        kwargs=(kwargs or {}))
    p.start()
    self._outstanding_subprocess_count += 1

  def _queue_to_list(self, queue_to_convert):
    """Convert `queue.Queue` to `list`."""
    list_to_return = []
    # Calling `queue.empty()` is not reliable.
    while True:
      try:
        list_to_return.append(queue_to_convert.get(block=False))
      except Queue.Empty:
        break
    return list_to_return

  def join(self, timeout=None):
    """Joins all the processes with timeout.

    Args:
      timeout: if set and not all processes report status within roughly
        `timeout` seconds, a `RuntimeError` exception will be thrown.

    Returns:
      It returns a tuple. The first element is a list that stores the return
      data added by subprocesses through `_add_return_data` or through normal
      function return; The second element is a list of the messages streamed to
      stdout and stderr in the subprocesses if `capture_std_stream` is True or
      `None` otherwise.

    Raises:
      RuntimeError: if not all processes report status within `timeout` seconds.
      Or the exception propagated from any child process.
    """
    if not timeout:
      if self._max_run_time:
        timeout = self._max_run_time + 10  # add 10 seconds grace period
      else:
        timeout = float('inf')
    start_time = time.time()
    while self._outstanding_subprocess_count > 0:
      while True:
        try:
          process_status = self._get_process_status_queue().get(timeout=10)
          break
        except Queue.Empty:
          if time.time() - start_time > timeout:
            # If none of those did, report timeout to user.
            raise RuntimeError(
                'One or more subprocesses timed out. Please use '
                '`--test_arg=--logtostderr` bazel flag to inspect logs for '
                'subprocess debugging info. Number of outstanding subprocesses '
                'is %d.' % self._outstanding_subprocess_count)

      self._outstanding_subprocess_count -= 1
      assert isinstance(process_status, _ProcessStatusInfo)
      if not process_status.is_successful:
        six.reraise(*process_status.exc_info)

    if self._capture_std_stream:
      # TODO(yuefengz): we need to make sure elements match the same process in
      # the two returned lists so as to not surprise users. Consider creating a
      # `ReturnData` class.
      return tuple(
          self._queue_to_list(multi_process_lib.get_user_data()[queue_name])
          for queue_name in [RETURN_VALUE_QUEUE, STD_STREAM_QUEUE])
    else:
      return (self._queue_to_list(
          multi_process_lib.get_user_data()[RETURN_VALUE_QUEUE]), None)

  def terminate(self, task_type, task_id):
    """Terminates the process with `task_type` and `task_id`."""
    self._get_parent_to_sub_queue().put('terminate {} {}'.format(
        task_type, task_id))

  def _add_return_data(self, data):
    """Adds return data that will be returned by `join`.

    The function provides a way for child processes to communicate with the
    parent process. Data passed to `_add_return_data` will be available in a
    Python Queue.Queue that is eventually returned by `join`.

    Args:
      data: data to be made available in the queue returned by `join`.
    """
    # TODO(rchao): Incorporate the task type and id information in a data
    # wrapper that becomes what is stored in the queue so we can tell where
    # the data is from.
    multi_process_lib.get_user_data()[RETURN_VALUE_QUEUE].put(data)

  def _add_std_stream_data_flattened(self, data):
    # TODO(yuefengz): currently the same queue is used by multiple processes. It
    # is difficult for users to distinguish between logs from different
    # processes.
    std_stream_queue = multi_process_lib.get_user_data()[STD_STREAM_QUEUE]
    for d in list(data):
      std_stream_queue.put(d)

  def _get_process_status_queue(self):
    return multi_process_lib.get_user_data()[PROCESS_STATUS_QUEUE]

  def _get_inter_process_queue(self):
    return multi_process_lib.get_user_data()[INTER_PROCESS_QUEUE]

  def _get_parent_to_sub_queue(self):
    return multi_process_lib.get_user_data()[PARENT_TO_SUB_QUEUE]


def run(proc_func,
        cluster_spec,
        max_run_time=None,
        capture_std_stream=False,
        grpc_fail_fast=False,
        timeout=None,
        args=None,
        kwargs=None):  # pylint: disable=g-doc-args
  """Runs functions in local child processes.

  It is a convenience method that creates a `MultiProcessRunner` object and
  invokes `start` and `join` method. Please see these methods for detailed
  documentations.

  Returns:
    A tuple returned from `MultiProcessRunner.join()`.
  """
  runner = MultiProcessRunner(
      proc_func,
      cluster_spec,
      max_run_time=max_run_time,
      capture_std_stream=capture_std_stream,
      grpc_fail_fast=grpc_fail_fast,
      args=args,
      kwargs=kwargs)
  runner.start()
  return runner.join(timeout)


def test_main():
  """Main function to be called within `__main__` of a test file."""
  with multi_process_lib.context_manager():
    test.main()
