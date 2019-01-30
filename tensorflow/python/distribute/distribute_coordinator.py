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
"""A component for running distributed TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import threading
import time

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
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
  STANDALONE_CLIENT = "standalone_client"

  # The distribute coordinator runs on each worker. It will run a standard
  # server on each worker and optionally run the `worker_fn` that is configured
  # to talk to its standard server.
  INDEPENDENT_WORKER = "independent_worker"


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
               strategy,
               cluster_spec,
               task_type,
               task_id,
               session_config=None,
               rpc_layer="grpc",
               worker_barrier=None):
    """Initialize the worker context object.

    Args:
      strategy: a `DistributionStrategy` object.
      cluster_spec: a ClusterSpec object. It can be empty or None in the local
        training case.
      task_type: a string indicating the role of the corresponding task, such as
        "worker" or "ps". It can be None if it is local training or in-graph
        replicated training.
      task_id: an integer indicating id of the corresponding task. It can be
        None if it is local training or in-graph replicated training.
      session_config: an optional `tf.ConfigProto` object.
      rpc_layer: optional string specifying the RPC protocol for communication
        with worker masters. If None or empty, hosts in the `cluster_spec` will
        be used directly.
      worker_barrier: optional, the barrier object for worker synchronization.
    """
    self._strategy = strategy
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._session_config = session_config
    self._worker_barrier = worker_barrier
    self._rpc_layer = rpc_layer
    self._master_target = self._get_master_target()
    self._num_workers = _get_num_workers(cluster_spec)
    self._is_chief_node = self._is_chief()

  def _debug_message(self):
    if self._cluster_spec:
      return "[cluster_spec: %r, task_type: %r, task_id: %r]" % (
          self._cluster_spec, self.task_type, self.task_id)
    else:
      return "[local]"

  def __enter__(self):
    old_context = distribute_coordinator_context.get_current_worker_context()
    if old_context:
      raise ValueError(
          "You cannot run distribute coordinator in a `worker_fn`.\t" +
          self._debug_message())
    # pylint: disable=protected-access
    distribute_coordinator_context._worker_context.current = self

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    # pylint: disable=protected-access
    distribute_coordinator_context._worker_context.current = None

  def _get_master_target(self):
    """Return the master target for a task."""
    # If cluster_spec is None or empty, we use local master.
    if not self._cluster_spec:
      return ""

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
      # TODO(yuefengz): we should throw an error in independent worker mode.
      return
    self._worker_barrier.wait()

  def session_creator(self,
                      scaffold=None,
                      config=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      max_wait_secs=7200):
    """Returns a session creator.

    The returned session creator will be configured with the correct master
    target and session configs. It will also run either init ops or ready ops
    by querying the `strategy` object when `create_session` is called on it.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string. Optional path to a directory where to restore
        variables.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
        Only one of `checkpoint_dir` or `checkpoint_filename_with_path` can be
        specified.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      a descendant of SessionCreator.
    """
    if config:
      session_config = copy.deepcopy(config)
      session_config.MergeFrom(self._session_config)
    else:
      session_config = self._session_config

    if not self._strategy or self._strategy.extended.experimental_should_init:
      logging.info("Creating chief session creator with config: %r", config)
      return monitored_session.ChiefSessionCreator(
          scaffold,
          master=self.master_target,
          config=session_config,
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=checkpoint_filename_with_path)
    else:
      logging.info("Creating worker session creator with config: %r", config)
      return monitored_session.WorkerSessionCreator(
          scaffold,
          master=self.master_target,
          config=session_config,
          max_wait_secs=max_wait_secs)

  @property
  def session_config(self):
    return copy.deepcopy(self._session_config)

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

  @property
  def experimental_should_init(self):
    """Whether to run init ops."""
    return self._strategy.extended.experimental_should_init

  @property
  def should_checkpoint(self):
    """Whether to save checkpoint."""
    return self._strategy.extended.should_checkpoint

  @property
  def should_save_summary(self):
    """Whether to save summaries."""
    return self._strategy.extended.should_save_summary


def _run_single_worker(worker_fn,
                       strategy,
                       cluster_spec,
                       task_type,
                       task_id,
                       session_config,
                       rpc_layer="",
                       worker_barrier=None,
                       coord=None):
  """Runs a single worker by calling `worker_fn` under context."""
  session_config = copy.deepcopy(session_config)
  strategy = copy.deepcopy(strategy)
  # If there is an EVALUATOR task, we run single-machine eval on that task.
  if task_type == _TaskType.EVALUATOR:
    # It is possible to not have a strategy object for EVALUATOR task.
    if strategy:
      strategy.configure(session_config)
  else:
    assert strategy
    strategy.configure(session_config, cluster_spec, task_type, task_id)

  context = _WorkerContext(
      strategy,
      cluster_spec,
      task_type,
      task_id,
      session_config=session_config,
      rpc_layer=rpc_layer,
      worker_barrier=worker_barrier)
  with context:
    if coord:
      with coord.stop_on_exception():
        return worker_fn(strategy)
    else:
      return worker_fn(strategy)


def _split_cluster_for_evaluator(cluster_spec, task_type):
  """Split the cluster for evaluator since it needn't talk to other tasks."""
  # Splitting the cluster is important to prevent the evaluator from talking to
  # other tasks in the cluster. Since we allow evaluator not to use
  # distribution strategies and as a result ops in the evalauator task may have
  # unspecified devices. Those ops may end up on other tasks if we don't split
  # the cluster.
  new_cluster_spec = multi_worker_util.normalize_cluster_spec(
      cluster_spec).as_dict()
  if task_type == _TaskType.EVALUATOR:
    assert _TaskType.EVALUATOR in new_cluster_spec
    new_cluster_spec = {
        _TaskType.EVALUATOR: new_cluster_spec[_TaskType.EVALUATOR]
    }
  else:
    new_cluster_spec.pop(_TaskType.EVALUATOR, None)
  return multi_worker_util.normalize_cluster_spec(new_cluster_spec)


def _run_std_server(cluster_spec=None,
                    task_type=None,
                    task_id=None,
                    session_config=None,
                    rpc_layer=None,
                    environment=None):
  """Runs a standard server."""
  assert cluster_spec
  target = cluster_spec.task_address(task_type, task_id)
  if rpc_layer:
    target = rpc_layer + "://" + target

  class _FakeServer(object):
    """A fake server that runs a master session."""

    def start(self):
      # A tensorflow server starts when a remote session is created.
      logging.info(
          "Creating a remote session to start a TensorFlow server, "
          "target = %r, session_config=%r", target, session_config)
      session.Session(target=target, config=session_config)

    def join(self):
      while True:
        time.sleep(5)

  if environment == "google":
    server = _FakeServer()
    server.start()
    return server
  else:
    if session_config:
      logging.info(
          "Starting standard TensorFlow server, target = %r, session_config= "
          "%r", target, session_config)
    else:
      logging.info("Starting standard TensorFlow server, target = %r", target)
    cluster_spec = _split_cluster_for_evaluator(cluster_spec, task_type)
    server = server_lib.Server(
        cluster_spec,
        job_name=task_type,
        task_index=task_id,
        config=session_config,
        protocol=rpc_layer)
    server.start()
    return server


def _run_between_graph_client(worker_fn, strategy, eval_fn, eval_strategy,
                              cluster_spec, session_config, rpc_layer):
  """Runs a standalone client for between-graph replication."""
  coord = coordinator.Coordinator()
  eval_thread = None
  if _TaskType.EVALUATOR in cluster_spec.jobs:
    eval_thread = threading.Thread(
        target=_run_single_worker,
        args=(eval_fn, eval_strategy, cluster_spec, _TaskType.EVALUATOR, 0,
              session_config),
        kwargs={
            "rpc_layer": rpc_layer,
            "coord": coord,
        })
    eval_thread.start()

  threads = []
  worker_barrier = _Barrier(_get_num_workers(cluster_spec))
  for task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
    for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
      t = threading.Thread(
          target=_run_single_worker,
          args=(worker_fn, strategy, cluster_spec, task_type, task_id,
                session_config),
          kwargs={
              "rpc_layer": rpc_layer,
              "worker_barrier": worker_barrier,
              "coord": coord,
          })
      t.start()
      threads.append(t)

  if eval_thread:
    # TODO(yuefengz): is it necessary to join eval thread?
    threads_to_join = threads + [eval_thread]
  else:
    threads_to_join = threads
  coord.join(threads_to_join)

  # TODO(yuefengz): we probably want to return results from all workers?
  return None


def _run_in_graph_client(worker_fn, strategy, eval_fn, eval_strategy,
                         cluster_spec, session_config, rpc_layer):
  """Runs a standalone client for in-graph replication."""
  coord = coordinator.Coordinator()
  eval_thread = None
  if _TaskType.EVALUATOR in cluster_spec.jobs:
    eval_thread = threading.Thread(
        target=_run_single_worker,
        args=(eval_fn, eval_strategy, cluster_spec, _TaskType.EVALUATOR, 0,
              session_config),
        kwargs={
            "rpc_layer": rpc_layer,
            "coord": coord,
        })
    eval_thread.start()

  worker_result = _run_single_worker(
      worker_fn,
      strategy,
      cluster_spec,
      None,
      None,
      session_config,
      rpc_layer=rpc_layer,
      coord=coord)

  if eval_thread:
    coord.join([eval_thread])

  return worker_result


def _configure_session_config_for_std_servers(
    strategy, eval_strategy, session_config, cluster_spec, task_type, task_id):
  # pylint: disable=g-doc-args
  """Call strategy's `configure` to mutate the session_config.

  The session_config is currently needed as default config for a TensorFlow
  server. In the future, we should be able to remove this method and only pass
  the session config to a client session.
  """
  if task_type == _TaskType.EVALUATOR:
    if eval_strategy:
      eval_strategy.configure(session_config=session_config)
  else:
    # The strategy may be shared in standalone client mode.
    strategy = copy.deepcopy(strategy)
    strategy.configure(
        session_config=session_config,
        cluster_spec=cluster_spec,
        task_type=task_type,
        task_id=task_id)
  # Remove the device filters specific to the strategy, so that the
  # TensorFlow server brought up with one strategy can be used by other
  # strategies. The device filters can be set in the client side as well.
  del session_config.device_filters[:]


def run_standard_tensorflow_server(session_config=None):
  """Starts a standard TensorFlow server.

  This method parses configurations from "TF_CONFIG" environment variable and
  starts a TensorFlow server. The "TF_CONFIG" is typically a json string and
  must have information of the cluster and the role of the server in the
  cluster. One example is:

  TF_CONFIG='{
      "cluster": {
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "worker", "index": 1}
  }'

  This "TF_CONFIG" specifies there are 3 workers and 2 ps tasks in the cluster
  and the current role is worker 1.

  Valid task types are "chief", "worker", "ps" and "evaluator" and you can have
  at most one "chief" and at most one "evaluator".

  An optional key-value can be specified is "rpc_layer". The default value is
  "grpc".

  Args:
    session_config: an optional `tf.ConfigProto` object. Users can pass in
      the session config object to configure server-local devices.

  Returns:
    a `tf.train.Server` object which has already been started.

  Raises:
    ValueError: if the "TF_CONFIG" environment is not complete.
  """
  tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
  if "cluster" not in tf_config:
    raise ValueError("\"cluster\" is not found in TF_CONFIG.")
  cluster_spec = multi_worker_util.normalize_cluster_spec(tf_config["cluster"])
  if "task" not in tf_config:
    raise ValueError("\"task\" is not found in TF_CONFIG.")
  task_env = tf_config["task"]
  if "type" not in task_env:
    raise ValueError(
        "\"task_type\" is not found in the `task` part of TF_CONFIG.")
  task_type = task_env["type"]
  task_id = int(task_env.get("index", 0))

  rpc_layer = tf_config.get("rpc_layer", "grpc")

  session_config = session_config or config_pb2.ConfigProto()
  # Set the collective group leader for collective ops to initialize collective
  # ops when server starts.
  if "chief" in cluster_spec.jobs:
    session_config.experimental.collective_group_leader = (
        "/job:chief/replica:0/task:0")
  else:
    if "worker" not in cluster_spec.jobs:
      raise ValueError(
          "You must have `chief` or `worker` jobs in the `cluster_spec`.")
    session_config.experimental.collective_group_leader = (
        "/job:worker/replica:0/task:0")

  server = _run_std_server(
      cluster_spec=cluster_spec,
      task_type=task_type,
      task_id=task_id,
      session_config=session_config,
      rpc_layer=rpc_layer)
  server.start()
  return server


# TODO(yuefengz): propagate cluster_spec in the STANDALONE_CLIENT mode.
# TODO(yuefengz): we may need a smart way to figure out whether the current task
# is the special task when we support cluster_spec propagation.
def run_distribute_coordinator(worker_fn,
                               strategy,
                               eval_fn=None,
                               eval_strategy=None,
                               mode=CoordinatorMode.STANDALONE_CLIENT,
                               cluster_spec=None,
                               task_type=None,
                               task_id=None,
                               session_config=None,
                               rpc_layer="grpc"):
  """Runs the coordinator for distributed TensorFlow.

  This function runs a split coordinator for distributed TensorFlow in its
  default mode, i.e the STANDALONE_CLIENT mode. Given a `cluster_spec`
  specifying server addresses and their roles in a cluster, this coordinator
  will figure out how to set them up, give the underlying function the right
  targets for master sessions via a scope object and coordinate their training.
  The cluster consisting of standard servers needs to be brought up either with
  the standard server binary or with a binary running distribute coordinator
  with `task_type` set to non-client type which will then turn into standard
  servers.

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

  The `strategy` object is expected to be a DistributionStrategy object which
  has implemented methods needed by distributed coordinator such as
  `configure(session_config, cluster_spec, task_type, task_id)` which configures
  the strategy object for a specific task and `experimental_should_init`
  property which instructs the distribute coordinator whether to run init ops
  for a task. The distribute coordinator will make a copy of the `strategy`
  object, call its `configure` method and pass it to `worker_fn` as an argument.

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

  For evaluation, if "evaluator" exists in the cluster_spec, a separate thread
  will be created to call `eval_fn` with its `task_type` set to "evaluator". If
  `eval_fn` is not defined, fall back to `worker_fn`. This implies that
  evaluation will be done on a single machine if there is an "evaluator" task.
  If "evaluator" doesn't exit in the cluster_spec, it entirely depends on the
  `worker_fn` for how to do evaluation.

  Args:
    worker_fn: the function to be called. The function should accept a
      `strategy` object and will be given access to a context object via a
      context manager scope.
    strategy: a DistributionStrategy object which specifying whether it should
      run between-graph replicated training or not, whether to run init ops,
      etc. This object will also be configured given `session_config`,
      `cluster_spec`, `task_type` and `task_id`.
    eval_fn: optional function for "evaluator" task. If `eval_fn` is not passed
      in but a "evaluator" task found in the `cluster_spec`, the `worker_fn`
      will be used for this task.
    eval_strategy: optional DistributionStrategy object for "evaluator" task.
    mode: in which mode this distribute coordinator runs.
    cluster_spec: a dict, ClusterDef or ClusterSpec specifying servers and roles
      in a cluster. If not set or empty, fall back to local training.
    task_type: the current task type, optional if this is a client.
    task_id: the current task id, optional if this is a client.
    session_config: an optional `tf.ConfigProto` object which will be passed
      to `strategy`'s `configure` method and used to create a session.
    rpc_layer: optional string, the protocol for RPC, e.g. "grpc".

  Raises:
    ValueError: if `cluster_spec` is supplied but not a dict or a ClusterDef or
      a ClusterSpec.

  Returns:
    In the client job, return the value returned by `worker_fn` if
    it is in-graph replication or INDEPENDENT_WORKER mode; return None
    otherwise.
  """
  tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
  if not cluster_spec:
    cluster_spec = tf_config.get("cluster", {})
    task_env = tf_config.get("task", {})
    if task_env:
      task_type = task_env.get("type", task_type)
      task_id = int(task_env.get("index", task_id))

  if cluster_spec:
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    # TODO(yuefengz): validate cluster_spec.

  rpc_layer = tf_config.get("rpc_layer", rpc_layer)
  environment = tf_config.get("environment", None)

  # Setting the session config is necessary for some strategies such
  # CollectiveAllReduceStrategy.
  session_config = session_config or config_pb2.ConfigProto(
      allow_soft_placement=True)

  if cluster_spec:
    logging.info(
        "Running Distribute Coordinator with mode = %r, cluster_spec = %r, "
        "task_type = %r, task_id = %r, environment = %r, rpc_layer = %r", mode,
        cluster_spec.as_dict(), task_type, task_id, environment, rpc_layer)

  if not cluster_spec:
    # `mode` is ignored in the local case.
    logging.info("Running local Distribute Coordinator.")
    _run_single_worker(worker_fn, strategy, None, None, None, session_config,
                       rpc_layer)
    if eval_fn:
      _run_single_worker(eval_fn, eval_strategy, None, None, None,
                         session_config, rpc_layer)
    else:
      logging.warning("Skipped evaluation since `eval_fn` is not passed in.")
  elif mode == CoordinatorMode.STANDALONE_CLIENT:
    if not eval_fn:
      logging.warning("`eval_fn` is not passed in. The `worker_fn` will be "
                      "used if an \"evaluator\" task exists in the cluster.")
    eval_fn = eval_fn or worker_fn
    if not eval_strategy:
      logging.warning("`eval_strategy` is not passed in. No distribution "
                      "strategy will be used for evaluation.")

    # The client must know the cluster but servers in the cluster don't have to
    # know the client.
    if task_type in [_TaskType.CLIENT, None]:
      if strategy.extended.experimental_between_graph:
        return _run_between_graph_client(worker_fn, strategy, eval_fn,
                                         eval_strategy, cluster_spec,
                                         session_config, rpc_layer)
      else:
        return _run_in_graph_client(worker_fn, strategy, eval_fn, eval_strategy,
                                    cluster_spec, session_config, rpc_layer)
    else:
      # If not a client job, run the standard server.
      _configure_session_config_for_std_servers(strategy, eval_strategy,
                                                session_config, cluster_spec,
                                                task_type, task_id)
      server = _run_std_server(
          cluster_spec=cluster_spec,
          task_type=task_type,
          task_id=task_id,
          session_config=session_config,
          rpc_layer=rpc_layer,
          environment=environment)
      server.join()
  else:
    if mode != CoordinatorMode.INDEPENDENT_WORKER:
      raise ValueError("Unexpected coordinator mode: %r" % mode)

    if not eval_fn:
      logging.warning("`eval_fn` is not passed in. The `worker_fn` will be "
                      "used if an \"evaluator\" task exists in the cluster.")
    eval_fn = eval_fn or worker_fn
    if not eval_strategy:
      logging.warning("`eval_strategy` is not passed in. No distribution "
                      "strategy will be used for evaluation.")

    # Every one starts a standard server, get session config from `configure`
    # method.
    _configure_session_config_for_std_servers(strategy, eval_strategy,
                                              session_config, cluster_spec,
                                              task_type, task_id)
    server = _run_std_server(
        cluster_spec=cluster_spec,
        task_type=task_type,
        task_id=task_id,
        session_config=session_config,
        rpc_layer=rpc_layer,
        environment=environment)

    if task_type in [_TaskType.CHIEF, _TaskType.WORKER]:
      if strategy.extended.experimental_between_graph:
        # All jobs run `worker_fn` if between-graph.
        return _run_single_worker(worker_fn, strategy, cluster_spec, task_type,
                                  task_id, session_config, rpc_layer)
      else:
        # Only one node runs `worker_fn` if in-graph.
        context = _WorkerContext(strategy, cluster_spec, task_type, task_id)
        if context.is_chief:
          return _run_single_worker(worker_fn, strategy, cluster_spec, None,
                                    None, session_config, rpc_layer)
        else:
          server.join()
    elif task_type == _TaskType.EVALUATOR:
      return _run_single_worker(eval_fn, eval_strategy, cluster_spec, task_type,
                                task_id, session_config, rpc_layer)
    else:
      if task_type != _TaskType.PS:
        raise ValueError("Unexpected task_type: %r" % task_type)
      server.join()
