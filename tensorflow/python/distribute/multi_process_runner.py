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
from absl import logging
import six
from six.moves import queue as Queue

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.platform import test

# _ProcessStatusInfo contains process status information. When is_successful
# attribute is True, the subprocess has ended successfully, or if False, the
# exception stack trace info is stored in exc_info to pass on to parent process
# to be re-raised.
_ProcessStatusInfo = collections.namedtuple(
    '_ProcessStatusInfo', ['task_type', 'is_successful', 'exc_info'])

# _SubprocessInfo collects basic information of a subprocess such as task type
# and process id.
# TODO(rchao): Include task_type and task_id in subprocess info.
_SubprocessInfo = collections.namedtuple('_SubprocessInfo', ['pid'])

# Information returned from a successful MultiProcessRunner run.
MultiProcessRunnerResult = collections.namedtuple('MultiProcessRunnerResult',
                                                  ['return_value', 'stdout'])

# Process status queue is used by `multi_process_runner` internally for
# communication from subprocesses to the parent process for whether it's been
# successful, and if not what the error stack trace is.
PROCESS_STATUS_QUEUE = 'process_status_queue'

# Return value queue is intended to be used by users of `multi_process_runner`
# for the process function to return information to the caller of
# `multi_process_runner.run()`.
RETURN_VALUE_QUEUE = 'return_value_queue'

# Subprocess info queue stores `_SubprocessInfo` for later potential
# termination by the parent.
SUBPROCESS_INFO_QUEUE = 'subprocess_info_queue'

# Parent-to-sub queue is used for communications from parent to subprocess.
# Currently this is only used to terminate subprocesses.
# TODO(rchao): Remove this once subprocess is terminated by SIGKILL.
PARENT_TO_SUB_QUEUE = 'parent_to_sub_queue'

# Streaming queue stores the logged and printed messages from subprocesses.
STREAMING_QUEUE = 'streaming_queue'

# Pipes to stream stdout and stderr from subprocesses to parent process.
STREAMING_PIPE = 'streaming_pipe'

# Barrier identifier.
BARRIER = 'barrier'

_DEFAULT_MAX_SUBPROCESS_COUNT = 20

# Next pipe index to be global so that pipes are not reused across multiple
# MultiProcessRunner usages.
# TODO(rchao): Investigate possibility to remove this variable.
_next_pipe_index = 0


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
               rpc_layer=None,
               max_run_time=None,
               grpc_fail_fast=None,
               stream_stdout=True,
               list_stdout=False,
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
      rpc_layer: RPC layer to use. Default value is 'grpc+loas'.
      max_run_time: If set, child processes is forced to exit at approximately
        this many seconds after `start` is called. We achieve this through
        `signal.alarm()` api. Note that this is best effort at Python level
        since Python signal handler does not get executed when it runs lower
        level C/C++ code. So it can be delayed for arbitrarily long time.
      grpc_fail_fast: Whether GRPC connection between processes should fail
        without retrying. Defaults to None, in which case the environment
        variable is not explicitly set.
      stream_stdout: True if the output/error from the subprocesses should be
        streamed to be printed in parent process' log. Defaults to True.
      list_stdout: True if the output/error from the subprocesses should be
        collected to be attached to the resulting `MultiProcessRunnerResult`
        returned from `MultiProcessRunner.join()`. If True, the list of stdout
        can be retrieved via `MultiProcessRunnerResult.stdout` attribute.
        Defaults to False.
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
    self._rpc_layer = rpc_layer
    self._max_run_time = max_run_time
    self._grpc_fail_fast = grpc_fail_fast
    self._stream_stdout = stream_stdout
    # TODO(rchao): Revisit list_stdout argument to consider other solution.
    self._list_stdout = list_stdout
    self._dependence_on_chief = True
    self._args = args or ()
    self._kwargs = kwargs or {}

    self._outstanding_subprocess_count = 0

    # Child processes should have the same v2 and eager behavior.
    self._v2_enabled = tf2.enabled()
    self._executing_eagerly = context.executing_eagerly()

    # This flag will be set to True once terminate_all() is called.
    self._all_forced_terminated = False

  def _continuously_readline_from_sub(self, pipe_r, task_type, task_id):
    """Function to continuously read lines from subprocesses."""
    reader = os.fdopen(pipe_r.fileno(), 'r')
    while True:
      read_line = reader.readline()
      if read_line == 'EOF':
        reader.close()
        # The thread that runs `_continuously_readline_from_sub` stops here.
        # However the threads don't exit until the test exits, so we do not
        # attempt to join the threads (which leads to timeout).
        # TODO(rchao): Understand why and do thread joining.
        break
      task_string = '[{}-{}]:'.format(task_type, task_id)
      formatted_line = '{} {}'.format(task_string.ljust(14), read_line)
      if self._stream_stdout:
        self._print_stdout_in_parent(formatted_line, task_type, task_id)
      if self._list_stdout:
        self._add_stdout_in_queue(formatted_line, task_type, task_id)

  def _print_stdout_in_parent(self, formatted_line, task_type, task_id):
    del task_type, task_id
    # Flush True so the logging order from subprocesses is respected.
    # TODO(rchao): Use a lock here to ensure the printed lines are not broken.
    print(formatted_line, end='', flush=True)

  def _add_stdout_in_queue(self, formatted_line, task_type, task_id):
    del task_type, task_id
    # A queue instead of a simple list is used here due to b/150652733.
    _resource(STREAMING_QUEUE).put(formatted_line)

  def _start_subprocess_and_reading_thread(self, proc_func, task_type, task_id,
                                           args, kwargs):
    """Start a subprocess and a thread the reads lines from the subprocess."""
    global _next_pipe_index
    pipe_r, pipe_w = _resource(STREAMING_PIPE)[_next_pipe_index]
    _next_pipe_index += 1

    p = multi_process_lib.Process(
        target=_Subprocess(),
        args=(proc_func, task_type, task_id, self._cluster_spec,
              self._rpc_layer, self._grpc_fail_fast, self._v2_enabled,
              self._executing_eagerly, pipe_w) + args,
        kwargs=kwargs)
    p.start()
    self._outstanding_subprocess_count += 1

    # For each subprocess, we dedicate a thread continuously reading lines
    # from them.
    thread = threading.Thread(  # pylint: disable=unexpected-keyword-arg
        target=self._continuously_readline_from_sub,
        args=(pipe_r, task_type, task_id))
    thread.start()

  def start(self):
    """Starts processes, one for each task in `cluster_spec`."""

    global _next_pipe_index
    self._starting_pipe_index = _next_pipe_index

    for task_type, addresses in self._cluster_spec.items():
      for task_id, _ in enumerate(addresses):
        self._start_subprocess_and_reading_thread(self._proc_func, task_type,
                                                  task_id, self._args,
                                                  self._kwargs)

    # TODO(rchao): Remove the need of using SIGALRM if possible. At this time,
    # without this the tests become very flaky.
    if self._max_run_time is not None:

      def handler(signum, frame):
        del signum, frame
        self.terminate_all()

      signal.signal(signal.SIGALRM, handler)
      signal.alarm(self._max_run_time)

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
    self._start_subprocess_and_reading_thread(proc_func, task_type, task_id,
                                              args or (), kwargs or {})

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
      A MultiProcessRunnerResult object, which has two attributes,
      `return_value` and `stdout`. `return_value` always contains the return
      values from the subprocesses. If `list_stdout` argument is True at
      `__init__`, `stdout` is available that contains a list of all messages
      from subprocesses' stdout and stderr.

    Raises:
      RuntimeError: if not all processes report status approximatelty within
      `timeout` seconds, or there's an exception propagated from any subprocess.
    """

    if not timeout:
      timeout = float('inf')
    start_time = time.time()
    while self._outstanding_subprocess_count > 0:
      while True:
        try:
          process_status = _resource(PROCESS_STATUS_QUEUE).get(timeout=10)
          break
        except Queue.Empty:
          if self._all_forced_terminated:
            break
          if time.time() - start_time > timeout:
            # If none of those did, report timeout to user.
            raise RuntimeError('One or more subprocesses timed out. '
                               'Number of outstanding subprocesses '
                               'is %d.' % self._outstanding_subprocess_count)

      if self._all_forced_terminated:
        break
      self._outstanding_subprocess_count -= 1
      assert isinstance(process_status, _ProcessStatusInfo)
      if not process_status.is_successful:
        six.reraise(*process_status.exc_info)

      if self._dependence_on_chief and process_status.task_type == 'chief':
        self.terminate_all()
        break

    # Giving threads some time to finish the message reading from subprocesses.
    time.sleep(5)

    stdout = self._queue_to_list(_resource(STREAMING_QUEUE))
    return_value = self._queue_to_list(_resource(RETURN_VALUE_QUEUE))

    # Notifying the threads that are reading lines that we should stop.
    for pipe_index in range(self._starting_pipe_index, _next_pipe_index):  # pylint: disable=protected-access
      _, pipe_w = _resource(STREAMING_PIPE)[pipe_index]
      writer = os.fdopen(pipe_w.fileno(), 'w')
      # Writing end of file message so the threads that's actively reading lines
      # know to stop.
      writer.writelines(['EOF'])
      writer.close()

    return MultiProcessRunnerResult(stdout=stdout, return_value=return_value)

  def terminate(self, task_type, task_id):
    """Terminates the process with `task_type` and `task_id`."""
    _resource(PARENT_TO_SUB_QUEUE).put('terminate {} {}'.format(
        task_type, task_id))

  def terminate_all(self):
    """Terminates all subprocesses."""
    subprocess_infos = []

    while True:
      try:
        subprocess_info = _resource(SUBPROCESS_INFO_QUEUE).get(block=False)
        subprocess_infos.append(subprocess_info)
      except Queue.Empty:
        break

    for subprocess_info in subprocess_infos:
      logging.info('Parent process is now killing PID: %d', subprocess_info.pid)
      os.kill(subprocess_info.pid, signal.SIGKILL)

    self._all_forced_terminated = True


class _Subprocess(object):
  """Represents an internal subprocess used in MultiProcessRunner's context."""

  @contextlib.contextmanager
  def _runtime_mode(self, executing_eagerly):
    if executing_eagerly:
      with context.eager_mode():
        yield
    else:
      with context.graph_mode():
        yield

  def _finish_process(self, process_status_info, return_value):
    """Adds data to queues before program exits."""
    # Clear the alarm.
    signal.alarm(0)

    if return_value is not None:
      self._add_return_data(return_value)
    _resource(PROCESS_STATUS_QUEUE).put(process_status_info)

  def _message_checking_func(self, task_type, task_id):
    """A function that regularly checks messages from parent process."""
    # TODO(rchao): Remove this once parent uses SIGKILL to terminate subprocess.
    while True:
      try:
        message = _resource(PARENT_TO_SUB_QUEUE).get(block=False)

        # Currently the only possible message is termination.
        if not message.startswith('terminate'):
          raise ValueError('Unrecognized message: {}'.format(message))

        if message == 'terminate {} {}'.format(task_type, task_id):
          break
        else:
          # If the message is not targeting this process, put it back to the
          # queue.
          _resource(PARENT_TO_SUB_QUEUE).put(message)
          time.sleep(1)
      except Queue.Empty:
        time.sleep(0.1)
    self._finish_process(
        _ProcessStatusInfo(
            task_type=task_type, is_successful=True, exc_info=None), None)
    # `os._exit(0)` is used to more reliably terminate a subprocess.
    os._exit(0)  # pylint: disable=protected-access

  def __call__(self, proc_func, task_type, task_id, per_process_cluster_spec,
               rpc_layer, grpc_fail_fast, v2_enabled, executing_eagerly, pipe_w,
               *arg, **kwargs):
    """The wrapper function that actually gets run in child process(es)."""

    pid = os.getpid()
    logging.info('Subprocess with PID %d is now being started.', pid)
    _resource(SUBPROCESS_INFO_QUEUE).put(_SubprocessInfo(pid=pid))

    # Assign sys.stdout and sys.stderr as duplicates of `pipe_w` so print() and
    # logging.*() write directly to `pipe_w`. Unfortunately since we cannot
    # prepend task_type and task_id information to the streamed logs we will
    # need a thread per subprocess to distinguish where the piece of message is
    # from.
    os.dup2(pipe_w.fileno(), sys.stdout.fileno())
    os.dup2(pipe_w.fileno(), sys.stderr.fileno())

    # The thread will be dedicated to checking messages from the parent process.
    threading.Thread(  # pylint: disable=unexpected-keyword-arg
        target=self._message_checking_func,
        args=(task_type, task_id),
        daemon=True).start()

    if grpc_fail_fast is not None:
      os.environ['GRPC_FAIL_FAST'] = str(grpc_fail_fast)
    tf_config_dict = {
        'cluster': per_process_cluster_spec,
        'task': {
            'type': task_type,
            'index': task_id,
        },
    }
    if rpc_layer is not None:
      tf_config_dict['rpc_layer'] = rpc_layer
    os.environ['TF_CONFIG'] = json.dumps(tf_config_dict)

    if v2_enabled:
      v2_compat.enable_v2_behavior()

    try:
      with self._runtime_mode(executing_eagerly):
        return_value = proc_func(*arg, **kwargs)
        is_successful = True
        exc_info = None

    except Exception:  # pylint: disable=broad-except
      # Capture all exceptions to be reported to parent process.
      return_value = None
      is_successful = False
      exc_info = sys.exc_info()

      # Re-raise the exception in addition to reporting it to the parent
      # process, so that even if `--test_timeout` flag is set and the
      # error doesn't make it to be shown in parent process before bazel's
      # timeout, the log would still show what happens in this subprocess,
      # instead of silently suppressing the error due to early bazel
      # timeout. Raising an error in the subprocess produces stack trace in
      # the log, but the program continues running.
      raise

    finally:
      self._finish_process(
          _ProcessStatusInfo(
              task_type=task_type,
              is_successful=is_successful,
              exc_info=exc_info),
          return_value)

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
    _resource(RETURN_VALUE_QUEUE).put(data)


def barrier():
  return multi_process_lib.get_user_data()[BARRIER]


def _resource(resource_name):
  return multi_process_lib.get_user_data()[resource_name]


def run(proc_func,
        cluster_spec,
        rpc_layer=None,
        max_run_time=None,
        grpc_fail_fast=None,
        stream_stdout=True,
        list_stdout=False,
        timeout=None,
        args=None,
        kwargs=None):  # pylint: disable=g-doc-args
  """Runs functions in local child processes.

  It is a convenience method that creates a `MultiProcessRunner` object and
  invokes `start` and `join` method. Please see these methods for detailed
  documentations.

  Returns:
    A MultiProcessRunnerResult object returned from `MultiProcessRunner.join()`.
  """
  runner = MultiProcessRunner(
      proc_func,
      cluster_spec,
      rpc_layer,
      max_run_time=max_run_time,
      grpc_fail_fast=grpc_fail_fast,
      stream_stdout=stream_stdout,
      list_stdout=list_stdout,
      args=args,
      kwargs=kwargs)
  runner.start()
  return runner.join(timeout)


def test_main(max_subprocess_count=_DEFAULT_MAX_SUBPROCESS_COUNT,
              barrier_parties=0):
  """Main function to be called within `__main__` of a test file.

  Args:
    max_subprocess_count: Maximum number of subprocesses that will be used. User
      of multi_process_runner needs to determine a number at calling this
      method, and the subprocesses involved later should not exceed this number.
    barrier_parties: Number of parties the barrier will be used toward. User of
      multi_process_runner needs to determine a number at calling this method.
  """
  with multi_process_lib.context_manager(max_subprocess_count, barrier_parties):
    test.main()
