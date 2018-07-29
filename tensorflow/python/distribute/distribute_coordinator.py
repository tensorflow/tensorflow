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


_coordinator_context = threading.local()


def get_current_coordinator_context():
  """Returns the current coordinator context."""
  try:
    return _coordinator_context.current
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


class _CoordinatorContext(object):
  """The coordinator context class.

  This context object provides configuration information for each task. One
  context manager with a coordinator context object will be created per
  invocation to the `worker_fn` where `get_current_coordinator_context` can be
  called to access the coordinator context object.
  """

  def __init__(self,
               cluster_spec,
               task_type,
               task_id,
               between_graph=False,
               rpc_layer="grpc",
               worker_barrier=None):
    """Initialize the coordinator context object.

    Args:
      cluster_spec: a ClusterSpec object. It can be empty or None in the local
        training case.
      task_type: a string indicating the role of the corresponding task, such as
        "worker" or "ps". It can be None if it is local training or
        `between_graph` is False.
      task_id: an integer indicating id of the corresponding task. It can be
        None if it is local training or `between_graph` is False.
      between_graph: whether it is between-graph replication or not.
      rpc_layer: optional string specifying the RPC protocol for communication
        with worker masters. If None or empty, hosts in the `cluster_spec` will
        be used directly.
      worker_barrier: optional, the barrier object for worker synchronization.

    Raises:
      ValueError: if task_type or task_id is Node or empty and it is distributed
        between-graph replicated training.
    """
    if cluster_spec and between_graph:
      if not task_type or task_id is None:
        raise ValueError("`task_type` and `task_id` must be set in the "
                         "distributed between-graph replicated training.")
      if task_type not in cluster_spec.jobs:
        raise ValueError("`task_type` %r not found in the `cluster_spec` %r" %
                         (task_type, cluster_spec))
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._worker_barrier = worker_barrier
    self._rpc_layer = rpc_layer
    self._master_target = self._get_master_target()
    self._num_workers = _get_num_workers(cluster_spec)
    self._is_chief_node = self._is_chief()

  def __enter__(self):
    old_context = get_current_coordinator_context()
    if old_context:
      raise ValueError(
          "You cannot run distribute coordinator in a `worker_fn`.")
    _coordinator_context.current = self

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    _coordinator_context.current = None

  def _get_master_target(self):
    """Return the master target for a task."""
    # If cluster_spec is None or empty, we use local master.
    if not self._cluster_spec:
      return "local"

    # If task_type is None, then it is in-graph replicated training. In this
    # case we use the chief or first worker's master target.
    if not self._task_type:
      if _TaskType.CHIEF in self._cluster_spec.jobs:
        assert not self.between_graph
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
    if (not self._cluster_spec or self._task_type in [_TaskType.CHIEF, None]):
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
      raise ValueError(
          "`worker_barrier is not set in the coordinator context.`")
    self._worker_barrier.wait()

  @property
  def distributed_mode(self):
    """Whether it is distributed training or not."""
    return bool(self._cluster_spec)

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


def _run(worker_fn, cluster_spec, task_type, task_id, between_graph, rpc_layer,
         worker_barrier):
  with _CoordinatorContext(cluster_spec, task_type, task_id, between_graph,
                           rpc_layer, worker_barrier):
    worker_fn()


def run_distribute_coordinator(worker_fn,
                               cluster_spec=None,
                               between_graph=False,
                               rpc_layer=None):
  """Run the coordinator for distributed TensorFlow.

  This function runs a unified and split coordinator for distributed TensorFlow.
  Given a `cluster_spec` specifying server addresses and their roles in a
  cluster, this coordinator will figure out how to set them up, give the
  underlying function the right targets for master sessions and coordinate their
  training.

  In addition to be the distribute coordinator, this is also the source of
  configurations for each job in the distributed training. As there are multiple
  ways to configure a distributed TensorFlow cluster, its context object
  provides these configurations so that users or higher-level APIs don't have to
  figure out the configuration for each job by themselves.

  In the between-graph replicated training, this coordinator will create
  multiple threads and each calls the `worker_fn` which is supposed to create
  its own graph and connect to one worker master given by its coordinator
  context. In the in-graph replicated training, it has only one thread calling
  this `worker_fn`.

  The `worker_fn` defines the training logic and is called under a its own
  coordinator context which can be accessed to via
  `get_current_coordinator_context`. A coordinator context provides access to
  configurations for each task, e.g. the task_type, task_id, master target and
  so on. Since `worker_fn` will be called in a thread and possibly multiple
  times, caller should be careful when it accesses global data. For example, it
  is unsafe to define flags in a `worker_fn` or to define different environment
  variables for different `worker_fn`s.

  The `worker_fn` for the between-graph replication is defined as if there are
  only one worker corresponding to the `worker_fn` and possibly ps jobs. It
  assigns variables to parameter servers and all other operations to that
  worker. In the in-graph replication case, the `worker_fn` has to define
  operations for all worker jobs. Using a distribution strategy can simplify the
  `worker_fn` by not having to worry about the replication and device assignment
  of variables and operations.

  This method is intended to be invoked by high-level APIs so that users don't
  have to explictly call it to run this coordinator. For those who don't use
  high-level APIs, to change a program to use this coordinator, wrap everything
  in a the program after global data definitions such as commandline flag
  definition into the `worker_fn` and get task-specific configurations from
  the coordinator context.

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
    cluster_spec: a dict, ClusterDef or ClusterSpec specifying servers and roles
      in a cluster. If not set or empty, fall back to local training.
    between_graph: a boolean. It is only useful when `cluster_spec` is set and
      not empty. If true, it will use between-graph replicated training;
      otherwise it will use in-graph replicated training.
    rpc_layer: optional string, the protocol for RPC, e.g. "grpc".

  Raises:
    ValueError: if `cluster_spec` is supplied but not a dict or a ClusterDef or
      a ClusterSpec.
  """
  if not cluster_spec:
    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    cluster_spec = tf_config.get("cluster", {})

  if cluster_spec:
    if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
      cluster_spec = server_lib.ClusterSpec(cluster_spec)
    elif not isinstance(cluster_spec, server_lib.ClusterSpec):
      raise ValueError(
          "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
          "`tf.train.ClusterDef` object")
    # TODO(yuefengz): validate cluster_spec.

  threads = []
  if cluster_spec and _TaskType.EVALUATOR in cluster_spec.jobs:
    t = threading.Thread(
        target=_run,
        args=(worker_fn, cluster_spec, _TaskType.EVALUATOR, 0, between_graph,
              rpc_layer, None))
    t.start()
    threads.append(t)

  if cluster_spec and between_graph:
    worker_barrier = _Barrier(_get_num_workers(cluster_spec))
    for task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
      for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
        t = threading.Thread(
            target=_run,
            args=(worker_fn, cluster_spec, task_type, task_id, between_graph,
                  rpc_layer, worker_barrier))
        t.start()
        threads.append(t)
  else:
    # Local or in-graph replicated training.
    _run(worker_fn, cluster_spec, None, None, between_graph, rpc_layer, None)

  # TODO(yuefengz): wrapper threads into thread coordinator?
  for t in threads:
    t.join()
