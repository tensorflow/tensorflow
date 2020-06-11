# Lint as: python3
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
import unittest

from absl import logging
import six
from six.moves import queue as Queue

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context

multiprocessing = multi_process_lib.multiprocessing

# pylint: disable=g-import-not-at-top
try:
  # `faulthandler` is not available in py2.
  import faulthandler
except ImportError:
  faulthandler = None

# TODO(b/150264776): Remove after resolving CI issue.
try:
  import dill
except ImportError:
  dill = None

# TODO(b/150264776): Remove after resolving CI issue.
try:
  import tblib.pickling_support
  # For pickling traceback objects.
  tblib.pickling_support.install()
except ImportError:
  pass


# _ProcessStatusInfo contains process status information. When is_successful
# attribute is True, the subprocess has ended successfully, or if False, the
# exception stack trace info is stored in exc_info to pass on to parent process
# to be re-raised.
_ProcessStatusInfo = collections.namedtuple(
    '_ProcessStatusInfo',
    ['task_type', 'is_successful', 'exc_info', 'return_value'])

# Information returned from a successful MultiProcessRunner run.
MultiProcessRunnerResult = collections.namedtuple('MultiProcessRunnerResult',
                                                  ['return_value', 'stdout'])

TestEnvironment = collections.namedtuple('TestEnvironment', [
    'task_type', 'task_id', 'cluster_spec', 'rpc_layer', 'grpc_fail_fast',
    'v2_enabled', 'executing_eagerly'
])

# Resources for communication between worker processes and the main process.
#
# `process_status_queue` is used by `multi_process_runner` internally for
#   communication from subprocesses to the parent process for whether it's been
#   successful, and if not what the error stack trace is.
# `parent_to_sub_queue` is used for communications from parent to subprocess.
#   Currently this is only used to terminate subprocesses.
# TODO(rchao): Remove this once subprocess is terminated by SIGKILL.
# `streaming_pipe_w` is to stream stdout and stderr from subprocesses to parent
#   process.
# `barrier` is a barrier for the party of all subprocesses.
Resources = collections.namedtuple('Resources', [
    'process_status_queue', 'parent_to_sub_queue', 'streaming_pipe_w', 'barrier'
])

# Default time out sec is selected so that it's handled before the default
# "medium" timeout of the test runs.
_DEFAULT_TIMEOUT_SEC = 200


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
               use_dill_for_args=True,
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
      use_dill_for_args: Whether to use dill to pickle `args` and `kwargs`. dill
        can pickle more objects, but doesn't work with types in
        `multiprocessing` library like `Mutex`.
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
    if not multi_process_lib.initialized():
      raise RuntimeError('`multi_process_runner` is not initialized. '
                         'Please call `multi_process_runner.test_main()` '
                         'within `if __name__ == \'__main__\':` block '
                         'in your python module to properly initialize '
                         '`multi_process_runner`.')
    if not callable(proc_func):
      raise ValueError('proc_func is not a callable')

    self._proc_func = proc_func
    self._cluster_spec = cluster_spec
    self._rpc_layer = rpc_layer
    self._max_run_time = max_run_time
    self._grpc_fail_fast = grpc_fail_fast
    self._stream_stdout = stream_stdout
    # TODO(rchao): Revisit list_stdout argument to consider other solution.
    self._list_stdout = list_stdout
    self._dependence_on_chief = True
    self._use_dill_for_args = use_dill_for_args
    self._args = args or ()
    self._kwargs = kwargs or {}

    # Child processes should have the same v2 and eager behavior.
    self._v2_enabled = tf2.enabled()
    self._executing_eagerly = context.executing_eagerly()

    self._joined = False
    self._processes = {}
    self._outstanding_subprocess_count = 0
    self._reading_threads = []

    self._manager = multiprocessing.Manager()
    self._process_status_queue = self._manager.Queue()
    self._parent_to_sub_queue = self._manager.Queue()
    parties = sum(len(addresses) for addresses in self._cluster_spec.values())
    self._barrier = self._manager.Barrier(parties)

    # We use a queue to collect outputs from worker processes since it's thread
    # safe.
    self._streaming_queue = self._manager.Queue()

    # This flag will be set to True once terminate_all() is called.
    self._all_forced_terminated = False

  def _continuously_readline_from_sub(self, pipe_r, task_type, task_id):
    """Function to continuously read lines from subprocesses."""
    with os.fdopen(pipe_r.fileno(), 'r', closefd=False) as reader:
      for line in reader:
        task_string = '[{}-{}]:'.format(task_type, task_id)
        formatted_line = '{} {}'.format(task_string.ljust(14), line)
        if self._stream_stdout:
          # TODO(rchao): Use a lock here to ensure the printed lines are not
          # broken.
          print(formatted_line, end='', flush=True)
        if self._list_stdout:
          self._streaming_queue.put(formatted_line)

  def _start_subprocess_and_reading_thread(self,
                                           task_type,
                                           task_id,
                                           cluster_spec=None,
                                           proc_func=None,
                                           args=None,
                                           kwargs=None):
    """Start a subprocess and a thread the reads lines from the subprocess."""

    if dill is None:
      raise unittest.SkipTest(
          'TODO(b/150264776): Resolve dependency issue in CI')

    test_env = TestEnvironment(
        task_type=task_type,
        task_id=task_id,
        cluster_spec=cluster_spec or self._cluster_spec,
        rpc_layer=self._rpc_layer,
        grpc_fail_fast=self._grpc_fail_fast,
        v2_enabled=self._v2_enabled,
        executing_eagerly=self._executing_eagerly,
    )
    pipe_r, pipe_w = multiprocessing.Pipe(duplex=False)
    resources = Resources(
        process_status_queue=self._process_status_queue,
        parent_to_sub_queue=self._parent_to_sub_queue,
        streaming_pipe_w=pipe_w,
        barrier=self._barrier,
    )
    if proc_func is None:
      proc_func, args, kwargs = self._proc_func, self._args, self._kwargs
    # Always use dill to pickle proc_func so that we support more callable
    # types, e.g. lambda.
    proc_func = dill.dumps(proc_func, dill.HIGHEST_PROTOCOL)
    if self._use_dill_for_args:
      args = dill.dumps(args, dill.HIGHEST_PROTOCOL)
      kwargs = dill.dumps(kwargs, dill.HIGHEST_PROTOCOL)

    p = _Process(
        test_env=test_env,
        target=_ProcFunc(),
        args=(resources, test_env, proc_func, args, kwargs,
              self._use_dill_for_args))
    p.start()
    self._processes[(task_type, task_id)] = p
    self._outstanding_subprocess_count += 1

    # For each subprocess, we dedicate a thread continuously reading lines
    # from them.
    thread = threading.Thread(  # pylint: disable=unexpected-keyword-arg
        target=self._continuously_readline_from_sub,
        args=(pipe_r, task_type, task_id))
    thread.start()
    self._reading_threads.append(thread)

  def start(self):
    """Starts processes, one for each task in `cluster_spec`."""
    if self._processes:
      raise ValueError('MultiProcessRunner already started.')
    for task_type, addresses in self._cluster_spec.items():
      for task_id, _ in enumerate(addresses):
        self._start_subprocess_and_reading_thread(task_type, task_id)

    # TODO(rchao): Remove the need of using SIGALRM if possible. At this time,
    # without this the tests become very flaky.
    if self._max_run_time is not None:

      def handler(signum, frame):
        del signum, frame
        self.terminate_all()

      signal.signal(signal.SIGALRM, handler)
      signal.alarm(self._max_run_time)

  def start_in_process_as(self, as_task_type, as_task_id):
    """Start the processes, with the specified task run in main process.

    This is similar to `start()` except that the task with task_type
    `as_task_type` and task_id `as_task_id` is run in the main process.
    This method is particularly useful when debugging tool such as `pdb` is
    needed in some specific task. Note that since this method is blocking until
    that specific task exits, additional actions would need a thread to be
    called:

    ```python
    def proc_func():
      # user code to be run
      import pdb; pdb.set_trace()

    def follow_ups():
      time.sleep(5)
      mpr.start_single_process(
          task_type='evaluator',
          task_id=0)

    mpr = multi_process_runner.MultiProcessRunner(
        proc_func,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1))
    threading.Thread(target=follow_ups).start()
    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
    mpr.join()
    ```

    Note that if `list_stdout=True`, the logs/stdout by task
    run by the main process is not available in result.stdout.

    Args:
      as_task_type: The task type to be run in the main process.
      as_task_id: The task id to be run in the main process.
    """
    if self._processes:
      raise ValueError('MultiProcessRunner already started.')
    for task_type, addresses in self._cluster_spec.items():
      for task_id, _ in enumerate(addresses):
        if not (task_type == as_task_type and task_id == as_task_id):
          self._start_subprocess_and_reading_thread(task_type, task_id)

    _set_tf_config(as_task_type, as_task_id, self._cluster_spec,
                   self._rpc_layer)
    self._proc_func(*self._args, **self._kwargs)

  def start_single_process(self,
                           task_type,
                           task_id,
                           cluster_spec=None,
                           proc_func=None,
                           args=None,
                           kwargs=None):
    """Starts a single process.

    This starts a process in the cluster with the task type, task id, and the
    process function (`proc_func`). If process function is `None`, the function
    provided at `__init__` will be used. If `cluster_spec` is `None`, the
    cluster spec provided at `__init__` will be used.

    TODO(rchao): It is meant that all subprocesses will be updated with the new
    cluster spec, but this has yet to be implemented. At this time only the
    newly started subprocess picks up this updated cluster spec.

    Args:
      task_type: The task type.
      task_id: The task id.
      cluster_spec: The cluster spec to be used on the newly started
        process. If `None`, the cluster spec provided at `__init__` will be
        used.
      proc_func: The process function to be run on the newly started
        process. If specified, specify `args` and `kwargs` as well. If `None`,
        the function provided at `__init__` will be used.
      args: Optional positional arguments to be supplied in `proc_func`.
      kwargs: Optional keyword arguments to be supplied in `proc_func`.
    """
    self._start_subprocess_and_reading_thread(
        task_type,
        task_id,
        cluster_spec=cluster_spec,
        proc_func=proc_func,
        args=args or (),
        kwargs=kwargs or {})

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

  def get_process_id(self, task_type, task_id):
    """Returns the subprocess id given the task type and task id."""
    p = self._processes.get((task_type, task_id), None)
    return p.pid if p else None

  def _join_or_terminate(self, task_type, task_id, process, timeout):
    """Joins a process. If it times out, terminate all procsses."""
    logging.info('joining %s-%d', task_type, task_id)
    process.join(timeout)
    # If exitcode is None, the process aren't terminated and this is a
    # timeout.
    if process.exitcode is None:
      # Force termination to dump worker processes stack trace.
      self.terminate_all(sig=signal.SIGTERM)
      raise RuntimeError('%s-%d and possibly more subprocesses timed out.' %
                         (task_type, task_id))

  def join(self, timeout=_DEFAULT_TIMEOUT_SEC):
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
    if self._joined:
      raise ValueError("MultiProcessRunner can't be joined twice.")
    self._joined = True

    chief = self._processes.get(('chief', 0), None)
    if self._dependence_on_chief and chief:
      self._join_or_terminate('chief', 0, chief, timeout)
      # Give other processes a chance to exit on their own.
      for p in self._processes.values():
        p.join(timeout=3)
      self.terminate_all()
    else:
      for (task_type, task_id), p in self._processes.items():
        self._join_or_terminate(task_type, task_id, p, timeout)

    for (task_type, task_id), p in self._processes.items():
      logging.info('%s-%d exit code: %s', task_type, task_id, p.exitcode)

    process_statuses = self._queue_to_list(self._process_status_queue)
    if not self._all_forced_terminated and len(
        process_statuses) != self._outstanding_subprocess_count:
      raise RuntimeError(
          'missing statuses from %d subproceses.' %
          (self._outstanding_subprocess_count - len(process_statuses)))
    return_values = []
    for process_status in process_statuses:
      assert isinstance(process_status, _ProcessStatusInfo)
      if not process_status.is_successful:
        six.reraise(*process_status.exc_info)
      if process_status.return_value is not None:
        return_values.append(process_status.return_value)

    logging.info('Joining log reading threads.')
    for thread in self._reading_threads:
      thread.join()
    logging.info('Joined log reading threads.')

    # Clear the alarm.
    signal.alarm(0)

    stdout = self._queue_to_list(self._streaming_queue)

    return MultiProcessRunnerResult(stdout=stdout, return_value=return_values)

  def terminate(self, task_type, task_id):
    """Terminates the process with `task_type` and `task_id`."""
    p = self._processes.get((task_type, task_id), None)
    if p is None:
      raise ValueError('{}-{} does not exist'.format(task_type, task_id))
    # TODO(crccw): change to use Process.terminate() as well.
    self._parent_to_sub_queue.put('terminate {} {}'.format(task_type, task_id))
    p.join()

  def terminate_all(self, sig=None):
    """Terminates all subprocesses."""
    # Use SIGKILL as default. In systems where that's unavailable such as
    # windows, use SIGTERM.
    sig = sig or getattr(signal, 'SIGKILL', signal.SIGTERM)
    for (task_type, task_id), p in self._processes.items():
      try:
        os.kill(p.pid, sig)
      except ProcessLookupError:
        logging.info('Attempting to kill %s-%d but it does not exist.',
                     task_type, task_id)
    self._all_forced_terminated = True


class _Process(multi_process_lib.Process):
  """A modified `multiprocessing.Process` that can set up environment variables."""

  # TODO(crccw): consider moving other logics in _ProcFunc to _Process.

  def __init__(self, test_env, **kwargs):
    super(_Process, self).__init__(**kwargs)
    self._test_env = test_env
    self._actual_run = getattr(self, 'run')
    self.run = self._run_with_setenv

  def _run_with_setenv(self):
    # We need to set environment variables before doing anything because
    # setenv() is not thread-safe.
    test_env = self._test_env
    if test_env.grpc_fail_fast is not None:
      os.environ['GRPC_FAIL_FAST'] = str(test_env.grpc_fail_fast)
    _set_tf_config(test_env.task_type, test_env.task_id, test_env.cluster_spec,
                   test_env.rpc_layer)
    return self._actual_run()


class _ProcFunc(object):
  """Represents a callable to run in a subprocess."""

  @contextlib.contextmanager
  def _runtime_mode(self, executing_eagerly):
    if executing_eagerly:
      with context.eager_mode():
        yield
    else:
      with context.graph_mode():
        yield

  def _message_checking_func(self, task_type, task_id):
    """A function that regularly checks messages from parent process."""
    # TODO(rchao): Remove this once parent uses SIGKILL to terminate subprocess.
    while True:
      try:
        message = self._resources.parent_to_sub_queue.get(block=False)

        # Currently the only possible message is termination.
        if not message.startswith('terminate'):
          raise ValueError('Unrecognized message: {}'.format(message))

        if message == 'terminate {} {}'.format(task_type, task_id):
          break
        else:
          # If the message is not targeting this process, put it back to the
          # queue.
          self._resources.parent_to_sub_queue.put(message)
          time.sleep(1)
      except Queue.Empty:
        time.sleep(0.1)
    self._resources.process_status_queue.put(
        _ProcessStatusInfo(
            task_type=task_type,
            is_successful=True,
            exc_info=None,
            return_value=None))
    # `os._exit(0)` is used to more reliably terminate a subprocess.
    os._exit(0)  # pylint: disable=protected-access

  def _close_streaming(self):
    """Close stdout, stderr and streaming pipe.

    We need to explicitly close them since Tensorflow may take a while to exit,
    so that the reading threads in the main process can exit more quickly.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.close()
    sys.stderr.close()
    self._resources.streaming_pipe_w.close()

  def __call__(self, resources, test_env, proc_func, args, kwargs,
               use_dill_for_args):
    """The wrapper function that actually gets run in child process(es)."""

    global _barrier

    self._resources = resources
    _barrier = self._resources.barrier
    proc_func = dill.loads(proc_func)
    if use_dill_for_args:
      args = dill.loads(args)
      kwargs = dill.loads(kwargs)

    if faulthandler is not None:
      faulthandler.enable()
      faulthandler.register(signal.SIGTERM, chain=True)

    # All logging should go to stderr to be streamed to the main process.
    logging.set_stderrthreshold(logging.DEBUG)

    # Assign sys.stdout and sys.stderr as duplicates of `streaming_pipe_w` so
    # print() and logging.*() write directly to `streaming_pipe_w`.
    # Unfortunately since we cannot prepend task_type and task_id information to
    # the streamed logs we will need a thread per subprocess to distinguish
    # where the piece of message is from.
    os.dup2(resources.streaming_pipe_w.fileno(), sys.stdout.fileno())
    os.dup2(resources.streaming_pipe_w.fileno(), sys.stderr.fileno())

    pid = os.getpid()
    logging.info('Subprocess with PID %d (%s, %d) is now being started.', pid,
                 test_env.task_type, test_env.task_id)

    # The thread will be dedicated to checking messages from the parent process.
    threading.Thread(  # pylint: disable=unexpected-keyword-arg
        target=self._message_checking_func,
        args=(test_env.task_type, test_env.task_id),
        daemon=True).start()

    if test_env.v2_enabled:
      v2_compat.enable_v2_behavior()

    try:
      with self._runtime_mode(test_env.executing_eagerly):
        return_value = proc_func(*args, **kwargs)
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
      info = _ProcessStatusInfo(
          task_type=test_env.task_type,
          is_successful=is_successful,
          exc_info=exc_info,
          return_value=return_value)
      self._resources.process_status_queue.put(info)
      self._close_streaming()


def _set_tf_config(task_type, task_id, cluster_spec, rpc_layer=None):
  """Set TF_CONFIG environment variable."""
  tf_config_dict = {
      'cluster': cluster_spec,
      'task': {
          'type': task_type,
          'index': task_id,
      },
  }
  if rpc_layer is not None:
    tf_config_dict['rpc_layer'] = rpc_layer
  os.environ['TF_CONFIG'] = json.dumps(tf_config_dict)


def run(proc_func,
        cluster_spec,
        rpc_layer=None,
        max_run_time=None,
        grpc_fail_fast=None,
        stream_stdout=True,
        list_stdout=False,
        timeout=_DEFAULT_TIMEOUT_SEC,
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


# This is set by MultiProcessRunner in worker processes.
_barrier = None


def barrier():
  if _barrier is None:
    raise ValueError(
        'barrier is not defined. It is likely because you are calling barrier()'
        'in the main process. barrier() can only be called in the subprocesses.'
    )
  return _barrier


def test_main():
  """Main function to be called within `__main__` of a test file."""
  multi_process_lib.test_main()
