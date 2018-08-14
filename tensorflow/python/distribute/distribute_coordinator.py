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
"""A unified and split coordinator for distributed TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import threading

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.training import server_lib


class _TaskType(object):
  PS = "ps"
  WORKER = "worker"
  CHIEF = "chief"
  EVALUATOR = "evaluator"
  CLIENT = "client"


# TODO(yuefengz): support another mode where the client colocates with one
# worker.
class CoordinatorMode(object):
  """Specify how distribute coordinator runs."""
  # The default mode where distribute coordinator will run as a standalone
  # client and connects to remote servers for training.  Each remote server can
  # use the distribute coordinator binary with task_type set correctly which
  # will then turn into standard servers.
  SPLIT_CLIENT = 0

  # The distribute coordinator runs on each worker. It will run a standard
  # server on each worker and optionally run the `worker_fn` that is configured
  # to talk to its standard server.
  INDEPENDENT_WORKER = 1


_worker_context = threading.local()


def get_current_worker_context():
  """Returns the current task context."""
  try:
    return _worker_context.current
  except AttributeError:
    return None


class _Barrier(object):
  """A reusable barrier class for worker synchronization."""

  def __init__(self, num_participants):
    """Initializes the barrier object.

    Args:
      num_participants: an integer which is the expected number of calls of
        `wait` pass to through this barrier.
    """
    self._num_participants = num_participants
    self._counter = 0
    self._flag = False
    self._local_sense = threading.local()
    self._lock = threading.Lock()
    self._condition = threading.Condition()

  def wait(self):
    """Waits until all other callers reach the same wait call."""
    if not hasattr(self._local_sense, "value"):
      self._local_sense.value = False
    self._local_sense.value = not self._flag
    with self._lock:
      self._counter += 1
      if self._counter == self._num_participants:
        self._counter = 0
        self._flag = self._local_sense.value
    with self._condition:
      while self._flag != self._local_sense.value:
        self._condition.wait()
      self._condition.notify_all()


def _get_num_workers(cluster_spec):
  """Gets number of workers including chief."""
  if not cluster_spec:
    return 0
  return len(cluster_spec.as_dict().get(_TaskType.WORKER, [])) + len(
      cluster_spec.as_dict().get(_TaskType.CHIEF, []))


class _WorkerContext(object):
  """The worker context class.

  This context object provides configuration information for each task. One
  context manager with a worker context object will be created per
  invocation to the `worker_fn` where `get_current_worker_context` can be called
  to access the worker context object.
  """

  def __init__(self,
               cluster_spec,
               task_type,
               task_id,
               rpc_layer="grpc",
               worker_barrier=None):
    """Initialize the worker context object.

    Args:
      cluster_spec: a ClusterSpec object. It can be empty or None in the local
        training case.
      task_type: a string indicating the role of the corresponding task, such as
        "worker" or "ps". It can be None if it is local training or in-graph
        replicated training.
      task_id: an integer indicating id of the corresponding task. It can be
        None if it is local training or in-graph replicated training.
      rpc_layer: optional string specifying the RPC protocol for communication
        with worker masters. If None or empty, hosts in the `cluster_spec` will
        be used directly.
      worker_barrier: optional, the barrier object for worker synchronization.
    """
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._worker_barrier = worker_barrier
    self._rpc_layer = rpc_layer
    self._master_target = self._get_master_target()
    self._num_workers = _get_num_workers(cluster_spec)
    self._is_chief_node = self._is_chief()

  def _debug_message(self):
    return "[cluster_spec: %r, task_type: %r, task_id: %r]" % (
        self._cluster_spec, self.task_type, self.task_id)

  def __enter__(self):
    old_context = get_current_worker_context()
    if old_context:
      raise ValueError(
          "You cannot run distribute coordinator in a `worker_fn`.\t" +
          self._debug_message())
    _worker_context.current = self

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    _worker_context.current = None

  def _get_master_target(self):
    """Return the master target for a task."""
    # If cluster_spec is None or empty, we use local master.
    if not self._cluster_spec:
      return "local"

    # If task_type is None, then it is in-graph replicated training. In this
    # case we use the chief or first worker's master target.
    if not self._task_type:
      if _TaskType.CHIEF in self._cluster_spec.jobs:
        task_type = _TaskType.CHIEF
        task_id = 0
      else:
        assert _TaskType.WORKER in self._cluster_spec.jobs
        task_type = _TaskType.WORKER
        task_id = 0
    else:
      task_type = self._task_type
      task_id = self._task_id

    prefix = ""
    if self._rpc_layer:
      prefix = self._rpc_layer + "://"
    return prefix + self._cluster_spec.job_tasks(task_type)[task_id or 0]

  def _is_chief(self):
    """Return whether the task is the chief worker."""
    if (not self._cluster_spec or
        self._task_type in [_TaskType.CHIEF, _TaskType.EVALUATOR, None]):
      return True

    # If not local and chief not in the cluster_spec, use the first worker as
    # chief.
    if (_TaskType.CHIEF not in self._cluster_spec.jobs and
        self._task_type == _TaskType.WORKER and self._task_id == 0):
      return True
    return False

  def wait_for_other_workers(self):
    """Waits for other workers to reach the same call to this method.

    Raises:
      ValueError: if `worker_barrier` is not passed to the __init__ method.
    """
    if not self._worker_barrier:
      raise ValueError("`worker_barrier is not set in the worker context.` \t" +
                       self._debug_message())
    self._worker_barrier.wait()

  @property
  def has_barrier(self):
    """Whether the barrier is set or not."""
    return self._worker_barrier is not None

  @property
  def distributed_mode(self):
    """Whether it is distributed training or not."""
    return bool(self._cluster_spec) and self._task_type != _TaskType.EVALUATOR

  @property
  def cluster_spec(self):
    """Returns a copy of the cluster_spec object."""
    return copy.deepcopy(self._cluster_spec)

  @property
  def task_type(self):
    """Returns the role of the corresponing task."""
    return self._task_type

  @property
  def task_id(self):
    """Returns the id or index of the corresponing task."""
    return self._task_id

  @property
  def master_target(self):
    """Returns the session master for the corresponding task to connect to."""
    return self._master_target

  @property
  def is_chief(self):
    """Returns whether the task is a chief node."""
    return self._is_chief_node

  @property
  def num_workers(self):
    """Returns number of workers in the cluster, including chief."""
    return self._num_workers


def _run_single_worker(worker_fn,
                       cluster_spec,
                       task_type,
                       task_id,
                       rpc_layer,
                       worker_barrier=None):
  """Runs a single worker by calling `worker_fn` under context."""
  with _WorkerContext(
      cluster_spec,
      task_type,
      task_id,
      rpc_layer=rpc_layer,
      worker_barrier=worker_barrier):
    worker_fn()


def _run_std_server(cluster_spec=None,
                    task_type=None,
                    task_id=None,
                    session_config=None,
                    rpc_layer=None):
  """Runs a standard server."""
  server = server_lib.Server(
      cluster_spec,
      job_name=task_type,
      task_index=task_id,
      config=session_config,
      protocol=rpc_layer)
  server.start()
  return server


def _run_between_graph_client(worker_fn, cluster_spec, rpc_layer):
  """Runs a standalone client for between-graph replication."""
  eval_thread = None
  if _TaskType.EVALUATOR in cluster_spec.jobs:
    eval_thread = threading.Thread(
        target=_run_single_worker,
        args=(worker_fn, cluster_spec, _TaskType.EVALUATOR, 0),
        kwargs={
            "rpc_layer": rpc_layer,
        })
    eval_thread.start()

  threads = []
  worker_barrier = _Barrier(_get_num_workers(cluster_spec))
  for task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
    for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
      t = threading.Thread(
          target=_run_single_worker,
          args=(worker_fn, cluster_spec, task_type, task_id),
          kwargs={
              "rpc_layer": rpc_layer,
              "worker_barrier": worker_barrier
          })
      t.start()
      threads.append(t)

  # TODO(yuefengz): wrap threads into thread coordinator?
  for t in threads:
    t.join()

  # TODO(yuefengz): is it necessary to join eval thread?
  if eval_thread:
    eval_thread.join()


def _run_in_graph_client(worker_fn, cluster_spec, rpc_layer):
  """Runs a standalone client for in-graph replication."""
  eval_thread = None
  if _TaskType.EVALUATOR in cluster_spec.jobs:
    eval_thread = threading.Thread(
        target=_run_single_worker,
        args=(worker_fn, cluster_spec, _TaskType.EVALUATOR, 0),
        kwargs={
            "rpc_layer": rpc_layer,
        })
    eval_thread.start()

  _run_single_worker(worker_fn, cluster_spec, None, None, rpc_layer)
  if eval_thread:
    eval_thread.join()


# TODO(yuefengz): propagate cluster_spec in the SPLIT_CLIENT mode.
# TODO(yuefengz): we may need a smart way to figure out whether the current task
# is the special task when we support cluster_spec propagation.
def run_distribute_coordinator(worker_fn,
                               mode=CoordinatorMode.SPLIT_CLIENT,
                               cluster_spec=None,
                               task_type=None,
                               task_id=None,
                               between_graph=False,
                               rpc_layer="grpc"):
  """Runs the coordinator for distributed TensorFlow.

  This function runs a split coordinator for distributed TensorFlow in its
  default mode, i.e the SPLIT_CLIENT mode. Given a `cluster_spec` specifying
  server addresses and their roles in a cluster, this coordinator will figure
  out how to set them up, give the underlying function the right targets for
  master sessions via a scope object and coordinate their training. The cluster
  consisting of standard servers needs to be brought up either with the standard
  server binary or with a binary running distribute coordinator with `task_type`
  set to non-client type which will then turn into standard servers.

  In addition to be the distribute coordinator, this is also the source of
  configurations for each job in the distributed training. As there are multiple
  ways to configure a distributed TensorFlow cluster, its context object
  provides these configurations so that users or higher-level APIs don't have to
  figure out the configuration for each job by themselves.

  In the between-graph replicated training, this coordinator will create
  multiple threads and each calls the `worker_fn` which is supposed to create
  its own graph and connect to one worker master given by its context object. In
  the in-graph replicated training, it has only one thread calling this
  `worker_fn`.

  Another mode is the INDEPENDENT_WORKER mode where each server runs a
  distribute coordinator which will start a standard server and optionally runs
  `worker_fn` depending whether it is between-graph training or in-graph
  replicated training.

  The `worker_fn` defines the training logic and is called under a its own
  worker context which can be accessed to via `get_current_worker_context`. A
  worker context provides access to configurations for each task, e.g. the
  task_type, task_id, master target and so on. Since `worker_fn` will be called
  in a thread and possibly multiple times, caller should be careful when it
  accesses global data. For example, it is unsafe to define flags in a
  `worker_fn` or to define different environment variables for different
  `worker_fn`s.

  The `worker_fn` for the between-graph replication is defined as if there is
  only one worker corresponding to the `worker_fn` and possibly ps jobs. For
  example, when training with parameter servers, it assigns variables to
  parameter servers and all other operations to that worker. In the in-graph
  replication case, the `worker_fn` has to define operations for all worker
  jobs. Using a distribution strategy can simplify the `worker_fn` by not having
  to worry about the replication and device assignment of variables and
  operations.

  This method is intended to be invoked by high-level APIs so that users don't
  have to explictly call it to run this coordinator. For those who don't use
  high-level APIs, to change a program to use this coordinator, wrap everything
  in a the program after global data definitions such as commandline flag
  definition into the `worker_fn` and get task-specific configurations from
  the worker context.

  The `cluster_spec` can be either passed by the argument or parsed from the
  "TF_CONFIG" envrionment variable. Example of a TF_CONFIG:
  ```
    cluster = {'chief': ['host0:2222'],
               'ps': ['host1:2222', 'host2:2222'],
               'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster})
  ```

  If `cluster_spec` is not given in any format, it becomes local training and
  this coordinator will connect to a local session.

  For evaluation, if "evaluator" exist in the cluster_spec, a separate thread
  will be created with its `task_type` set to "evaluator". If "evaluator" is not
  set in the cluster_spec, it entirely depends on the `worker_fn` for how to do
  evaluation.

  Args:
    worker_fn: the function to be called and given the access to a coordinator
      context object.
    mode: in which mode this distribute coordinator runs.
    cluster_spec: a dict, ClusterDef or ClusterSpec specifying servers and roles
      in a cluster. If not set or empty, fall back to local training.
    task_type: the current task type, optional if this is a client.
    task_id: the current task id, optional if this is a client.
    between_graph: a boolean. It is only useful when `cluster_spec` is set and
      not empty. If true, it will use between-graph replicated training;
      otherwise it will use in-graph replicated training.
    rpc_layer: optional string, the protocol for RPC, e.g. "grpc".

  Raises:
    ValueError: if `cluster_spec` is supplied but not a dict or a ClusterDef or
      a ClusterSpec.
  """
  tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
  if not cluster_spec:
    cluster_spec = tf_config.get("cluster", {})
    task_env = tf_config.get("task", {})
    if task_env:
      task_type = task_env.get("type", task_type)
      task_id = int(task_env.get("index", task_id))

  if cluster_spec:
    if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
      cluster_spec = server_lib.ClusterSpec(cluster_spec)
    elif not isinstance(cluster_spec, server_lib.ClusterSpec):
      raise ValueError(
          "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
          "`tf.train.ClusterDef` object")
    # TODO(yuefengz): validate cluster_spec.

  if not cluster_spec:
    # `mode` is ignored in the local case.
    _run_single_worker(worker_fn, None, None, None, rpc_layer)
  elif mode == CoordinatorMode.SPLIT_CLIENT:
    # The client must know the cluster but servers in the cluster don't have to
    # know the client.
    if task_type in [_TaskType.CLIENT, None]:
      if between_graph:
        _run_between_graph_client(worker_fn, cluster_spec, rpc_layer)
      else:
        _run_in_graph_client(worker_fn, cluster_spec, rpc_layer)
    else:
      # If not a client job, run the standard server.
      server = _run_std_server(
          cluster_spec=cluster_spec, task_type=task_type, task_id=task_id)
      server.join()
  else:
    if mode != CoordinatorMode.INDEPENDENT_WORKER:
      raise ValueError("Unexpected coordinator mode: %r" % mode)

    # Every one starts a standard server.
    server = _run_std_server(
        cluster_spec=cluster_spec, task_type=task_type, task_id=task_id)

    if task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
      if between_graph:
        # All jobs run `worker_fn` if between-graph.
        _run_single_worker(worker_fn, cluster_spec, task_type, task_id,
                           rpc_layer)
      else:
        # Only one node runs `worker_fn` if in-graph.
        context = _WorkerContext(cluster_spec, task_type, task_id, rpc_layer)
        if context.is_chief:
          _run_single_worker(worker_fn, cluster_spec, None, None, rpc_layer)
        else:
          server.join()
    elif task_type == _TaskType.EVALUATOR:
      _run_single_worker(worker_fn, cluster_spec, task_type, task_id, rpc_layer)
    else:
      if task_type != _TaskType.PS:
        raise ValueError("Unexpected task_type: %r" % task_type)
      server.join()
