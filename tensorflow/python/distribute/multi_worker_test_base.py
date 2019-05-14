# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Base testing class for strategies that require multiple nodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import numpy as np

_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:  # pylint: disable=invalid-name
  _portpicker_import_error = _error
  portpicker = None

# pylint: disable=g-import-not-at-top
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.eager import context
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest


original_run_std_server = dc._run_std_server  # pylint: disable=protected-access

ASSIGNED_PORTS = set()
lock = threading.Lock()


def pick_unused_port():
  """Returns an unused and unassigned local port."""
  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type

  global ASSIGNED_PORTS
  with lock:
    while True:
      port = portpicker.pick_unused_port()
      if port > 10000 and port not in ASSIGNED_PORTS:
        ASSIGNED_PORTS.add(port)
        logging.info('Using local port %r', port)
        return port


def _create_cluster(num_workers,
                    num_ps,
                    has_chief=False,
                    has_eval=False,
                    protocol='grpc',
                    worker_config=None,
                    ps_config=None,
                    eval_config=None):
  """Creates and starts local servers and returns the cluster_spec dict."""
  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type
  worker_ports = [pick_unused_port() for _ in range(num_workers)]
  ps_ports = [pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  if num_workers > 0:
    cluster_dict['worker'] = ['localhost:%s' % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict['ps'] = ['localhost:%s' % port for port in ps_ports]
  if has_eval:
    cluster_dict['evaluator'] = ['localhost:%s' % pick_unused_port()]
  if has_chief:
    cluster_dict['chief'] = ['localhost:%s' % pick_unused_port()]

  cs = server_lib.ClusterSpec(cluster_dict)

  for i in range(num_workers):
    server_lib.Server(
        cs,
        job_name='worker',
        protocol=protocol,
        task_index=i,
        config=worker_config,
        start=True)

  for i in range(num_ps):
    server_lib.Server(
        cs,
        job_name='ps',
        protocol=protocol,
        task_index=i,
        config=ps_config,
        start=True)

  if has_chief:
    server_lib.Server(
        cs,
        job_name='chief',
        protocol=protocol,
        task_index=0,
        config=worker_config,
        start=True)

  if has_eval:
    server_lib.Server(
        cs,
        job_name='evaluator',
        protocol=protocol,
        task_index=0,
        config=eval_config,
        start=True)

  return cluster_dict


def create_in_process_cluster(num_workers,
                              num_ps,
                              has_chief=False,
                              has_eval=False):
  """Create an in-process cluster that consists of only standard server."""
  # Leave some memory for cuda runtime.
  gpu_mem_frac = 0.7 / (num_workers + int(has_chief) + int(has_eval))
  worker_config = config_pb2.ConfigProto()
  worker_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac

  # Enable collective ops which has no impact on non-collective ops.
  # TODO(yuefengz, tucker): removing this after we move the initialization of
  # collective mgr to the session level.
  if has_chief:
    worker_config.experimental.collective_group_leader = (
        '/job:chief/replica:0/task:0')
  else:
    worker_config.experimental.collective_group_leader = (
        '/job:worker/replica:0/task:0')

  ps_config = config_pb2.ConfigProto()
  ps_config.device_count['GPU'] = 0

  eval_config = config_pb2.ConfigProto()
  eval_config.experimental.collective_group_leader = ''

  # Create in-process servers. Once an in-process tensorflow server is created,
  # there is no way to terminate it. So we create one cluster per test process.
  # We could've started the server in another process, we could then kill that
  # process to terminate the server. The reasons why we don't want multiple
  # processes are
  # 1) it is more difficult to manage these processes;
  # 2) there is something global in CUDA such that if we initialize CUDA in the
  # parent process, the child process cannot initialize it again and thus cannot
  # use GPUs (https://stackoverflow.com/questions/22950047).
  return _create_cluster(
      num_workers,
      num_ps=num_ps,
      has_chief=has_chief,
      has_eval=has_eval,
      worker_config=worker_config,
      ps_config=ps_config,
      eval_config=eval_config,
      protocol='grpc')


def create_cluster_spec(has_chief=False,
                        num_workers=1,
                        num_ps=0,
                        has_eval=False):
  """Create a cluster spec with tasks with unused local ports."""
  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type

  cluster_spec = {}
  if has_chief:
    cluster_spec['chief'] = ['localhost:%s' % pick_unused_port()]
  if num_workers:
    cluster_spec['worker'] = [
        'localhost:%s' % pick_unused_port() for _ in range(num_workers)
    ]
  if num_ps:
    cluster_spec['ps'] = [
        'localhost:%s' % pick_unused_port() for _ in range(num_ps)
    ]
  if has_eval:
    cluster_spec['evaluator'] = ['localhost:%s' % pick_unused_port()]
  return cluster_spec


class MultiWorkerTestBase(test.TestCase):
  """Base class for testing multi node strategy and dataset."""

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    cls._cluster_spec = create_in_process_cluster(num_workers=2, num_ps=1)
    cls._default_target = 'grpc://' + cls._cluster_spec['worker'][0]

  def setUp(self):
    # We only cache the session in one test because another test may have a
    # different session config or master target.
    self._thread_local = threading.local()
    self._thread_local.cached_session = None
    self._result = 0
    self._lock = threading.Lock()

  @contextlib.contextmanager
  def session(self, graph=None, config=None, target=None):
    """Create a test session with master target set to the testing cluster.

    Creates a test session that connects to the local testing cluster.

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      target: the target of session to connect to.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    config = self._create_config(config)

    if target is None:
      target = self._default_target
    with session.Session(graph=graph, config=config, target=target) as sess:
      yield sess

  @contextlib.contextmanager
  # TODO(b/117573461): Overwrite self.evaluate() to use this function.
  def cached_session(self, graph=None, config=None, target=None):
    """Create a test session with master target set to the testing cluster.

    Creates a test session that connects to the local testing cluster.
    The session is only created once per test and then reused.

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      target: the target of session to connect to.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case. Note that the
      session will live until the end of the test.
    """
    config = self._create_config(config)

    if target is None:
      target = self._default_target
    if getattr(self._thread_local, 'cached_session', None) is None:
      self._thread_local.cached_session = session.Session(
          graph=None, config=config, target=target)
    sess = self._thread_local.cached_session
    with sess.graph.as_default(), sess.as_default():
      yield sess

  def _create_config(self, config):
    if config is None:
      config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      config = copy.deepcopy(config)
    # Don't perform optimizations for tests so we don't inadvertently run
    # gpu ops on cpu
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF)

    return config

  def _run_client(self, client_fn, task_type, task_id, num_gpus, eager_mode,
                  *args, **kwargs):
    if eager_mode:
      with context.eager_mode():
        result = client_fn(task_type, task_id, num_gpus, *args, **kwargs)
    else:
      with context.graph_mode():
        result = client_fn(task_type, task_id, num_gpus, *args, **kwargs)
    if np.all(result):
      with self._lock:
        self._result += 1

  def _run_between_graph_clients(self, client_fn, cluster_spec, num_gpus, *args,
                                 **kwargs):
    """Runs several clients for between-graph replication.

    Args:
      client_fn: a function that needs to accept `task_type`, `task_id`,
        `num_gpus` and returns True if it succeeds.
      cluster_spec: a dict specifying jobs in a cluster.
      num_gpus: number of GPUs per worker.
      *args: will be passed to `client_fn`.
      **kwargs: will be passed to `client_fn`.
    """
    threads = []
    for task_type in [run_config.TaskType.CHIEF, run_config.TaskType.WORKER]:
      for task_id in range(len(cluster_spec.get(task_type, []))):
        t = threading.Thread(
            target=self._run_client,
            args=(client_fn, task_type, task_id, num_gpus,
                  context.executing_eagerly()) + args,
            kwargs=kwargs)
        t.start()
        threads.append(t)
    for t in threads:
      t.join()
    self.assertEqual(self._result, len(threads))


class MockOsEnv(collections.Mapping):
  """A class that allows per-thread TF_CONFIG."""

  def __init__(self, *args):
    self._dict = dict()
    self._thread_local = threading.local()
    super(MockOsEnv, self).__init__(*args)

  def get(self, key, default=None):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    if key == 'TF_CONFIG':
      return dict.get(self._thread_local.dict, key, default)
    else:
      return dict.get(self._dict, key, default)

  def __getitem__(self, key):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    if key == 'TF_CONFIG':
      return dict.__getitem__(self._thread_local.dict, key)
    else:
      return dict.__getitem__(self._dict, key)

  def __setitem__(self, key, val):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    if key == 'TF_CONFIG':
      return dict.__setitem__(self._thread_local.dict, key, val)
    else:
      return dict.__setitem__(self._dict, key, val)

  def __iter__(self):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    for x in self._thread_local.dict:
      yield x
    for x in self._dict:
      yield x

  def __len__(self):
    if not hasattr(self._thread_local, 'dict'):
      self._thread_local.dict = dict()
    return self._thread_local.dict.__len__() + self._dict.__len__()


class IndependentWorkerTestBase(test.TestCase):
  """Testing infra for independent workers."""

  def _make_mock_run_std_server(self):

    def _mock_run_std_server(*args, **kwargs):
      ret = original_run_std_server(*args, **kwargs)
      # Wait for all std servers to be brought up in order to reduce the chance
      # of remote sessions taking local ports that have been assigned to std
      # servers. Only call this barrier the first time this function is run for
      # each thread.
      if not getattr(self._thread_local, 'server_started', False):
        self._barrier.wait()
      self._thread_local.server_started = True
      return ret

    return _mock_run_std_server

  def setUp(self):
    self._mock_os_env = MockOsEnv()
    self._mock_context = test.mock.patch.object(os, 'environ',
                                                self._mock_os_env)
    self._coord = coordinator.Coordinator()
    super(IndependentWorkerTestBase, self).setUp()
    self._mock_context.__enter__()
    # threading local object to be shared by all threads
    self._thread_local = threading.local()

  def tearDown(self):
    self._mock_context.__exit__(None, None, None)
    super(IndependentWorkerTestBase, self).tearDown()

  def _task_thread(self, task_fn, tf_config, executing_eagerly, *args,
                   **kwargs):
    with self._coord.stop_on_exception():
      os.environ['TF_CONFIG'] = json.dumps(tf_config)
      # Force the new thread simulating a worker to run in the same context
      # mode as the parent thread does.
      if executing_eagerly:
        with context.eager_mode():
          task_fn(*args, **kwargs)
      else:
        with ops.Graph().as_default(), context.graph_mode():
          task_fn(*args, **kwargs)

  def _run_task_in_thread(self, task_fn, cluster_spec, task_type, task_id,
                          *args, **kwargs):
    """Run tasks in a thread.

    If `tf_config` is provided, use it for the new thread; if not, construct one
    from `cluster_spec`, `task_type`, and `task_id`, and provide it to the new
    thread to be set as `TF_CONFIG` environment.

    Arguments:
      task_fn: The function to run in the new thread.
      cluster_spec: The cluster spec.
      task_type: The task type.
      task_id: The task id.
      *args: Additional positional arguments to provide to the thread's task_fn.
      **kwargs: Additional keyword arguments to provide to the thread's task_fn.
        If `tf_config` is provided, that dict will be used for the TF_CONFIG for
        the new thread.

    Returns:
      The thread that has started.
    """
    tf_config = kwargs.pop('tf_config', None)
    if tf_config is None:
      if task_type:
        tf_config = {
            'cluster': cluster_spec,
            'task': {
                'type': task_type,
                'index': task_id
            }
        }
      else:
        tf_config = {
            'cluster': cluster_spec,
        }
    t = threading.Thread(
        target=self._task_thread,
        args=(task_fn, tf_config, context.executing_eagerly()) + args,
        kwargs=kwargs)
    t.start()
    return t

  def run_multiple_tasks_in_threads(self, task_fn, cluster_spec, *args,
                                    **kwargs):
    # The task_fn should create std_server by itself.
    threads = {}
    for task_type in cluster_spec.keys():
      threads[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        t = self._run_task_in_thread(task_fn, cluster_spec, task_type, task_id,
                                     *args, **kwargs)
        threads[task_type].append(t)
    return threads

  def join_independent_workers(self, worker_threads):
    try:
      self._coord.join(worker_threads)
    except errors.UnknownError as e:
      if 'Could not start gRPC server' in e.message:
        self.skipTest('Cannot start std servers.')
      else:
        raise


class MultiWorkerMultiProcessTest(test.TestCase):
  """Testing infra for independent workers using multiple processes."""

  def _run_task_in_process(self, cmd_args, cluster_spec, task_type, task_id):
    env = os.environ.copy()
    env['TF_CONFIG'] = json.dumps({
        'cluster': cluster_spec,
        'task': {
            'type': task_type,
            'index': task_id
        }
    })
    return subprocess.Popen(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

  def run_multiple_tasks_in_processes(self, cmd_args, cluster_spec):
    """Run `cmd_args` in a process for each task in `cluster_spec`."""
    processes = {}
    for task_type in cluster_spec.keys():
      processes[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        p = self._run_task_in_process(cmd_args, cluster_spec, task_type,
                                      task_id)
        processes[task_type].append(p)
    return processes

  def join_independent_workers(self, worker_processes):
    return_codes = []
    for p in nest.flatten(worker_processes):
      try:
        # Calling p.wait() will hang if we don't consume its output.
        p.communicate()
      except ValueError:
        # The output of the process may have been consumed, in which case
        # calling `p.communicate()` will raise a ValueError.
        pass
      finally:
        return_codes.append(p.returncode)
    for return_code in return_codes:
      self.assertEqual(return_code, 0)

  def stream_stderr(self, process):
    # TODO(yuefengz): calling stream_stderr on a single process will probably
    # make all processes hang if they have too much output e.g. adding
    # --vmodule=execute=2 to cmd_args. But this method is useful for debugging
    # purposes. We should figure out the hanging problem, probably by consuming
    # outputs of all processes at the same time.
    while True:
      output = process.stderr.readline()
      if not output and process.poll() is not None:
        break
      if output:
        print(output.strip())
        sys.stdout.flush()


def get_tf_config_task():
  return json.loads(os.environ['TF_CONFIG'])['task']


def get_tf_config_cluster_spec():
  return json.loads(os.environ['TF_CONFIG'])['cluster']


def get_task_type():
  return get_tf_config_task()['type']


def get_task_index():
  return get_tf_config_task()['index']


def is_chief():
  return ('chief' not in get_tf_config_cluster_spec()
          and get_task_type() == 'worker'
          and get_task_index() == 0)
