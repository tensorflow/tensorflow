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

from absl import flags
import six
from six.moves import queue as Queue

from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.platform import test

_FINISH_PROPERLY_MESSAGE = 'OK'
_ExcInfoWrapper = collections.namedtuple('_ExcInfoWrapper', ['exc_info'])


class _AvailableQueues(object):
  """Names of the available queues used by `multi_process_runner`."""
  # Internal queue is used by `multi_process_runner` internally for
  # communication from subprocesses to the parent process. The message
  # can be _FINISH_PROPERLY_MESSAGE in which case the subprocess has ended successfully, or
  # the detailed message of an exception if the subprocess has raised
  # one so it can be re-raised by the parent process.
  INTERNAL_QUEUE = 'internal_queue'
  # Public queue is intended to be used by users of `multi_process_runner`
  # for the process function to return information to the caller of
  # `multi_process_runner.run()`.
  PUBLIC_QUEUE = 'public_queue'
  # Standard stream queue is used by `multi_process_runner` to collect
  # information streamed to stdout and stderr to be reported back to the
  # parent process.
  STD_STREAM_QUEUE = 'std_stream_queue'


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
  """A utility to start multiple subprocesses to simulate multiple workers.

  Training with multiple workers with eager runtime can be tested by simulating
  using multiple processes. See `run()` for more information about the usage
  of this class.
  """

  def run(self,
          proc_func,
          cluster_spec,
          proc_flags=None,
          timeout=200,
          time_to_exit=None,
          return_std_stream=False,
          args=None,
          kwargs=None):
    """Run functions on local sub-processes.

    Experimental. API subject to change. To fully inspect logging from
    subprocesses, use `--test_arg=--logtostderr` flag with bazel test.

    Args:
      proc_func: Function to be run on the processes. This will be run on
        processes for all task types.
      cluster_spec: Dict for cluster spec. The following is an example of
        cluster with three workers and two ps's.
        {"worker": ["worker0.example.com:2222",
                    "worker1.example.com:2222",
                    "worker2.example.com:2222"],
         "ps": ["ps0.example.com:2222",
                "ps1.example.com:2222"]}
      proc_flags: Dict that contains the key/values of the flags used on the
        processes.
      timeout: Time out in seconds. If the sub-process takes more than this time
        to complete, raise an error.
      time_to_exit: If set, sub-processes is forced to exit at approximately
        this many seconds after `run()` is called, through `signal.alarm()` api.
        This is for simulation of interruption on a process so in such cases no
        error is raised. Note that this is best effort at Python level since
        Python signal handler does not get executed inside the low-level (C)
        signal handler, so it can be delayed.
      return_std_stream: Boolean, whether the messages streamed to stdout and
        stderr in subprocesses are captured. If True, the messages are stored in
        a list returned as the second element.
      args: Positional arguments to be sent to functions run on processes.
      kwargs: Keyword arguments to be sent to functions run on processes.

    Returns:
      If `return_std_stream` is False, a list that stores the return data added
      by subprocesses through `multi_process_runner._add_return_data(data)`
      call,
      or through normal function return; if `return_std_stream` is True, a
      two-element tuple of `(return_data_list, std_stream_data_list)`, where
      `return_data_list` stores the return data added by processes through
      `multi_process_runner._add_return_data(data)` call or through normal
      function
      return, and `std_stream_data_list` stores the messages streamed to stdout
      and stderr in the subprocesses.

    Raises:
      RuntimeError: If any of the subprocesses raise an error, or if any of the
        subprocesses does not return or error out within `timeout` seconds.
    """

    assert cluster_spec is not None
    assert callable(proc_func)

    if not multi_process_lib.using_context_manager():
      raise RuntimeError('`multi_process_runner` is not initialized. '
                         'Please call `multi_process_runner.test_main()` '
                         'within `if __name__ == \'__main__\':` block '
                         'in your python module to properly initialize '
                         '`multi_process_runner`.')

    processes = []
    args = args or ()
    kwargs = kwargs or {}

    def wrapper_func(tf_config_as_json, proc_func, proc_flags, time_to_exit,
                     executing_eagerly, *arg, **kwargs):
      """The wrapper function that actually gets run on the process(es)."""

      @contextlib.contextmanager
      def runtime_mode(executing_eagerly):
        if executing_eagerly:
          with context.eager_mode():
            yield
        else:
          with context.graph_mode():
            yield

      with runtime_mode(executing_eagerly):
        os.environ['TF_CONFIG'] = tf_config_as_json
        if proc_flags is not None:
          for flag_key, flag_value in proc_flags.items():
            setattr(flags.FLAGS, flag_key, flag_value)

        stdout_collector = _LogCollector(
            sys.__stdout__) if return_std_stream else None
        stderr_collector = _LogCollector(
            sys.__stderr__) if return_std_stream else None

        def finish_wrapper_func_properly(func_result):
          """Call to finish `wrapper_func` properly."""
          # Clear the alarm.
          signal.alarm(0)
          if (return_std_stream and stdout_collector is not None and
              stderr_collector is not None):
            # If stdout and stderr are to be collected, add them to std stream
            # queue.
            self._add_std_stream_data_flattened(stdout_collector.log)
            self._add_std_stream_data_flattened(stderr_collector.log)
            # Un-redirect stdout and stderr.
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
          self._get_internal_queue().put(func_result)

        if time_to_exit is not None:

          def handler(signum, frame):
            del signum, frame
            finish_wrapper_func_properly(_FINISH_PROPERLY_MESSAGE)
            # pylint: disable=protected-access
            os._exit(0)

          signal.signal(signal.SIGALRM, handler)
          signal.alarm(time_to_exit)

        if return_std_stream:
          sys.stdout = stdout_collector
          sys.stderr = stderr_collector

        try:
          return_data = proc_func(*arg, **kwargs)
          if return_data is not None:
            self._add_return_data(return_data)
        # pylint: disable=broad-except
        except Exception:
          # Capture all exceptions to be reported to parent process.
          finish_wrapper_func_properly(_ExcInfoWrapper(sys.exc_info()))

          # Re-raise the exception in addition to reporting it to the parent
          # process, so that even if `--test_timeout` flag is set and the
          # error doesn't make it to be shown in parent process before bazel's
          # timeout, the log would still show what happens in this subprocess,
          # instead of silently suppressing the error due to early bazel
          # timeout. Raising an error in the subprocess produces stack trace in
          # the log, but the program continues running.
          raise

        finish_wrapper_func_properly(_FINISH_PROPERLY_MESSAGE)

    # Start number of processes according to `count_dict`.
    for job_type, addresses in cluster_spec.items():
      for task_id, _ in enumerate(addresses):
        tf_config_as_json = json.dumps({
            'cluster': cluster_spec,
            'task': {
                'type': job_type,
                'index': task_id
            }
        })
        p = multi_process_lib.Process(
            target=wrapper_func,
            args=(tf_config_as_json, proc_func, proc_flags, time_to_exit,
                  context.executing_eagerly()) + args,
            kwargs=kwargs)
        p.start()
        processes.append(p)

    internal_queue_results = []
    for _ in range(len(processes)):
      try:
        internal_queue_results.append(
            self._get_internal_queue().get(timeout=timeout))
      except Queue.Empty:
        # First check if any of the subprocesses raised exception.
        for internal_queue_result in internal_queue_results:
          if isinstance(internal_queue_result, _ExcInfoWrapper):
            six.reraise(*internal_queue_result.exc_info)
        # If none of those did, report time out to user.
        raise RuntimeError(
            'One or more subprocesses timed out. Please use '
            '`--test_arg=--logtostderr` bazel flag to inspect logs for '
            'subprocess debugging info. Timeout = {} sec.'.format(timeout))

    for internal_queue_result in internal_queue_results:
      if isinstance(internal_queue_result, _ExcInfoWrapper):
        six.reraise(*internal_queue_result.exc_info)
      assert internal_queue_result == _FINISH_PROPERLY_MESSAGE

    def queue_to_list(queue_to_convert):
      """Convert `queue.Queue` to `list`."""
      list_to_return = []
      while True:
        try:
          list_to_return.append(queue_to_convert.get(block=False))
        except Queue.Empty:
          break
      return list_to_return

    if return_std_stream:
      return tuple(
          queue_to_list(multi_process_lib.get_user_data()[queue_name])
          for queue_name in
          [_AvailableQueues.PUBLIC_QUEUE, _AvailableQueues.STD_STREAM_QUEUE])
    else:
      return queue_to_list(
          multi_process_lib.get_user_data()[_AvailableQueues.PUBLIC_QUEUE])

  def _add_return_data(self, data):
    """Add return data that will be returned by `multi_process_runner.run()`.

    The function provides a way for processes started by
    `multi_process_runner.run()` to communicate with the original process
    that started the sub-processes. Data passed to `_add_return_data` will
    be available in a python Queue.Queue that is eventually returned by
    `multi_process_runner.run()`.

    Args:
      data: data to be made available in the queue returned by
        `multi_process_runner.run()`.
    """
    # TODO(rchao): Incorporate the task type and id information in a data
    # wrapper that becomes what is stored in the queue so we can tell where
    # the data is from.
    multi_process_lib.get_user_data()[_AvailableQueues.PUBLIC_QUEUE].put(data)

  def _add_std_stream_data_flattened(self, data):
    std_stream_queue = multi_process_lib.get_user_data()[
        _AvailableQueues.STD_STREAM_QUEUE]
    for d in list(data):
      std_stream_queue.put(d)

  def _get_internal_queue(self):
    return multi_process_lib.get_user_data()[_AvailableQueues.INTERNAL_QUEUE]


def test_main():
  """Main function to be called within `__main__` of a test file."""
  with multi_process_lib.context_manager():
    test.main()


def job_count_to_cluster_spec(job_count_dict):
  """Convert a job count dict to cluster spec.

  Args:
    job_count_dict: Dict for task_type/count of such task type.
        {'worker': 1, 'ps': 1} is an example of a cluster with a worker and a
          ps.

  Returns:
    The converted cluster spec dict.
  """

  cluster_spec = {}
  for task_type, count in job_count_dict.items():
    cluster_spec[task_type] = [
        'localhost:{}'.format(multi_worker_test_base.pick_unused_port())
        for _ in range(count)
    ]
  return cluster_spec
