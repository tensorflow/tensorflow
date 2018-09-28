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

import contextlib
import copy
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
from tensorflow.python.estimator import run_config
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


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
                    ps_config=None):
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
        config=worker_config,
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
    cls._cluster_spec = create_in_process_cluster(num_workers=2, num_ps=0)
    cls._default_target = 'grpc://' + cls._cluster_spec['worker'][0]

  def setUp(self):
    # We only cache the session in one test because another test may have a
    # different session config or master target.
    self._thread_local = threading.local()
    self._thread_local.cached_session = None
    self._result = 0
    self._lock = threading.Lock()

  @contextlib.contextmanager
  def test_session(self, graph=None, config=None, target=None):
    """Create a test session with master target set to the testing cluster.

    This overrides the base class' method, removes arguments that are not needed
    by the multi-node case and creates a test session that connects to the local
    testing cluster.

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      target: the target of session to connect to.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if self.id().endswith('.test_session'):
      self.skipTest('Not a test.')

    if config is None:
      config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      config = copy.deepcopy(config)
    # Don't perform optimizations for tests so we don't inadvertently run
    # gpu ops on cpu
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF)

    if target is None:
      target = self._default_target
    if graph is None:
      if getattr(self._thread_local, 'cached_session', None) is None:
        self._thread_local.cached_session = session.Session(
            graph=None, config=config, target=target)
      sess = self._thread_local.cached_session
      with sess.graph.as_default(), sess.as_default():
        yield sess
    else:
      with session.Session(graph=graph, config=config, target=target) as sess:
        yield sess

  def _run_client(self, client_fn, task_type, task_id, num_gpus, *args,
                  **kwargs):
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
            args=(client_fn, task_type, task_id, num_gpus) + args,
            kwargs=kwargs)
        t.start()
        threads.append(t)
    for t in threads:
      t.join()
    self.assertEqual(self._result, len(threads))
