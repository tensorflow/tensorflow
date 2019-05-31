# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Training helper that checkpoints models and computes summaries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.Supervisor"])
class Supervisor(object):
  """A training helper that checkpoints models and computes summaries.

  This class is deprecated. Please use
  `tf.compat.v1.train.MonitoredTrainingSession` instead.

  The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
  and a `SessionManager` that takes care of common needs of TensorFlow
  training programs.

  #### Use for a single program

  ```python
  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
    sv = Supervisor(logdir='/tmp/mydir')
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session(FLAGS.master) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  Within the `with sv.managed_session()` block all variables in the graph have
  been initialized.  In addition, a few services have been started to
  checkpoint the model and add summaries to the event log.

  If the program crashes and is restarted, the managed session automatically
  reinitialize variables from the most recent checkpoint.

  The supervisor is notified of any exception raised by one of the services.
  After an exception is raised, `should_stop()` returns `True`.  In that case
  the training loop should also stop.  This is why the training loop has to
  check for `sv.should_stop()`.

  Exceptions that indicate that the training inputs have been exhausted,
  `tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
  but are not re-raised from the `with` block: they indicate a normal
  termination.

  #### Use for multiple replicas

  To train with replicas you deploy the same program in a `Cluster`.
  One of the tasks must be identified as the *chief*: the task that handles
  initialization, checkpoints, summaries, and recovery.  The other tasks
  depend on the *chief* for these services.

  The only change you have to do to the single program code is to indicate
  if the program is running as the *chief*.

  ```python
  # Choose a task as the chief. This could be based on server_def.task_index,
  # or job_def.name, or job_def.tasks. It's entirely up to the end user.
  # But there can be only one *chief*.
  is_chief = (server_def.task_index == 0)
  server = tf.distribute.Server(server_def)

  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that uses log directory on a shared file system.
    # Indicate if you are the 'chief'
    sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
    # Get a Session in a TensorFlow server on the cluster.
    with sv.managed_session(server.target) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  In the *chief* task, the `Supervisor` works exactly as in the first example
  above.  In the other tasks `sv.managed_session()` waits for the Model to have
  been initialized before returning a session to the training code.  The
  non-chief tasks depend on the chief task for initializing the model.

  If one of the tasks crashes and restarts, `managed_session()`
  checks if the Model is initialized.  If yes, it just creates a session and
  returns it to the training code that proceeds normally.  If the model needs
  to be initialized, the chief task takes care of reinitializing it; the other
  tasks just wait for the model to have been initialized.

  NOTE: This modified program still works fine as a single program.
  The single program marks itself as the chief.

  #### What `master` string to use

  Whether you are running on your machine or in the cluster you can use the
  following values for the --master flag:

  * Specifying `''` requests an in-process session that does not use RPC.

  * Specifying `'local'` requests a session that uses the RPC-based
    "Master interface" to run TensorFlow programs. See
    `tf.train.Server.create_local_server` for
    details.

  * Specifying `'grpc://hostname:port'` requests a session that uses
    the RPC interface to a specific host, and also allows the in-process
    master to access remote tensorflow workers. Often, it is
    appropriate to pass `server.target` (for some `tf.distribute.Server`
    named `server).

  #### Advanced use

  ##### Launching additional services

  `managed_session()` launches the Checkpoint and Summary services (threads).
  If you need more services to run you can simply launch them in the block
  controlled by `managed_session()`.

  Example: Start a thread to print losses.  We want this thread to run
  every 60 seconds, so we launch it with `sv.loop()`.

  ```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess, ))
    while not sv.should_stop():
      sess.run(my_train_op)
  ```

  ##### Launching fewer services

  `managed_session()` launches the "summary" and "checkpoint" threads which use
  either the optionally `summary_op` and `saver` passed to the constructor, or
  default ones created automatically by the supervisor.  If you want to run
  your own summary and checkpointing logic, disable these services by passing
  `None` to the `summary_op` and `saver` parameters.

  Example: Create summaries manually every 100 steps in the chief.

  ```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
  ```

  ##### Custom model initialization

  `managed_session()` only supports initializing the model by running an
  `init_op` or restoring from the latest checkpoint.  If you have special
  initialization needs, see how to specify a `local_init_op` when creating the
  supervisor.  You can also use the `SessionManager` directly to create a
  session and check if it could be initialized automatically.
  """

  # Value to pass for the 'ready_op', 'init_op', 'summary_op', 'saver',
  # and 'global_step' parameters of Supervisor.__init__() to indicate that
  # the default behavior should be used.
  USE_DEFAULT = 0

  @deprecation.deprecated(None,
                          "Please switch to tf.train.MonitoredTrainingSession")
  def __init__(self,
               graph=None,
               ready_op=USE_DEFAULT,
               ready_for_local_init_op=USE_DEFAULT,
               is_chief=True,
               init_op=USE_DEFAULT,
               init_feed_dict=None,
               local_init_op=USE_DEFAULT,
               logdir=None,
               summary_op=USE_DEFAULT,
               saver=USE_DEFAULT,
               global_step=USE_DEFAULT,
               save_summaries_secs=120,
               save_model_secs=600,
               recovery_wait_secs=30,
               stop_grace_secs=120,
               checkpoint_basename="model.ckpt",
               session_manager=None,
               summary_writer=USE_DEFAULT,
               init_fn=None,
               local_init_run_options=None):
    """Create a `Supervisor`.

    Args:
      graph: A `Graph`.  The graph that the model will use.  Defaults to the
        default `Graph`.  The supervisor may add operations to the graph before
        creating a session, but the graph should not be modified by the caller
        after passing it to the supervisor.
      ready_op: 1-D string `Tensor`.  This tensor is evaluated by supervisors in
        `prepare_or_wait_for_session()` to check if the model is ready to use.
        The model is considered ready if it returns an empty array.  Defaults to
        the tensor returned from `tf.compat.v1.report_uninitialized_variables()`
        If `None`, the model is not checked for readiness.
      ready_for_local_init_op: 1-D string `Tensor`.  This tensor is evaluated by
        supervisors in `prepare_or_wait_for_session()` to check if the model is
        ready to run the local_init_op. The model is considered ready if it
        returns an empty array. Defaults to `None`. If `None`, the model is not
        checked for readiness before running local_init_op.
      is_chief: If True, create a chief supervisor in charge of initializing and
        restoring the model.  If False, create a supervisor that relies on a
        chief supervisor for inits and restore.
      init_op: `Operation`.  Used by chief supervisors to initialize the model
        when it can not be recovered.  Defaults to an `Operation` that
        initializes all global variables.  If `None`, no initialization is done
        automatically unless you pass a value for `init_fn`, see below.
      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
        This feed dictionary will be used when `init_op` is evaluated.
      local_init_op: `Operation`. Used by all supervisors to run initializations
        that should run for every new supervisor instance. By default these are
        table initializers and initializers for local variables. If `None`, no
        further per supervisor-instance initialization is done automatically.
      logdir: A string.  Optional path to a directory where to checkpoint the
        model and log events for the visualizer.  Used by chief supervisors. The
        directory will be created if it does not exist.
      summary_op: An `Operation` that returns a Summary for the event logs. Used
        by chief supervisors if a `logdir` was specified.  Defaults to the
        operation returned from summary.merge_all().  If `None`, summaries are
        not computed automatically.
      saver: A Saver object.  Used by chief supervisors if a `logdir` was
        specified.  Defaults to the saved returned by Saver(). If `None`, the
        model is not saved automatically.
      global_step: An integer Tensor of size 1 that counts steps.  The value
        from 'global_step' is used in summaries and checkpoint filenames.
        Default to the op named 'global_step' in the graph if it exists, is of
        rank 1, size 1, and of type tf.int32 or tf.int64.  If `None` the global
        step is not recorded in summaries and checkpoint files.  Used by chief
        supervisors if a `logdir` was specified.
      save_summaries_secs: Number of seconds between the computation of
        summaries for the event log.  Defaults to 120 seconds.  Pass 0 to
        disable summaries.
      save_model_secs: Number of seconds between the creation of model
        checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.
      recovery_wait_secs: Number of seconds between checks that the model is
        ready.  Used by supervisors when waiting for a chief supervisor to
        initialize or restore the model.  Defaults to 30 seconds.
      stop_grace_secs: Grace period, in seconds, given to running threads to
        stop when `stop()` is called.  Defaults to 120 seconds.
      checkpoint_basename: The basename for checkpoint saving.
      session_manager: `SessionManager`, which manages Session creation and
        recovery. If it is `None`, a default `SessionManager` will be created
        with the set of arguments passed in for backwards compatibility.
      summary_writer: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None` to
        indicate that no summaries should be written.
      init_fn: Optional callable used to initialize the model. Called after the
        optional `init_op` is called.  The callable must accept one argument,
        the session being initialized.
      local_init_run_options: RunOptions to be passed as the SessionManager
        local_init_run_options parameter.

    Returns:
      A `Supervisor`.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
    `Supervisor`s are not supported when eager execution is enabled.
    @end_compatibility
    """
    if context.executing_eagerly():
      raise RuntimeError("Supervisors are compatible with eager execution.")
    # Set default values of arguments.
    if graph is None:
      graph = ops.get_default_graph()
    with graph.as_default():
      self._init_ready_op(
          ready_op=ready_op, ready_for_local_init_op=ready_for_local_init_op)
      self._init_init_op(init_op=init_op, init_feed_dict=init_feed_dict)
      self._init_local_init_op(local_init_op=local_init_op)
      self._init_saver(saver=saver)
      self._init_summary_op(summary_op=summary_op)
      self._init_global_step(global_step=global_step)
    self._graph = graph
    self._meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=graph.as_graph_def(add_shapes=True),
        saver_def=self._saver.saver_def if self._saver else None)
    self._is_chief = is_chief
    self._coord = coordinator.Coordinator()
    self._recovery_wait_secs = recovery_wait_secs
    self._stop_grace_secs = stop_grace_secs
    self._init_fn = init_fn
    self._local_init_run_options = local_init_run_options

    # Set all attributes related to checkpointing and writing events to None.
    # Afterwards, set them appropriately for chief supervisors, as these are
    # the only supervisors that can write checkpoints and events.
    self._logdir = None
    self._save_summaries_secs = None
    self._save_model_secs = None
    self._save_path = None
    self._summary_writer = None

    if self._is_chief:
      self._logdir = logdir
      self._save_summaries_secs = save_summaries_secs
      self._save_model_secs = save_model_secs
      if self._logdir:
        self._save_path = os.path.join(self._logdir, checkpoint_basename)
      if summary_writer is Supervisor.USE_DEFAULT:
        if self._logdir:
          self._summary_writer = _summary.FileWriter(self._logdir)
      else:
        self._summary_writer = summary_writer
      self._graph_added_to_summary = False

    self._init_session_manager(session_manager=session_manager)
    self._verify_setup()
    # The graph is not allowed to change anymore.
    graph.finalize()

  def _init_session_manager(self, session_manager=None):
    if session_manager is None:
      self._session_manager = session_manager_mod.SessionManager(
          local_init_op=self._local_init_op,
          ready_op=self._ready_op,
          ready_for_local_init_op=self._ready_for_local_init_op,
          graph=self._graph,
          recovery_wait_secs=self._recovery_wait_secs,
          local_init_run_options=self._local_init_run_options)
    else:
      self._session_manager = session_manager

  def _get_first_op_from_collection(self, key):
    """Returns the first `Operation` from a collection.

    Args:
      key: A string collection key.

    Returns:
      The first Op found in a collection, or `None` if the collection is empty.
    """
    try:
      op_list = ops.get_collection(key)
      if len(op_list) > 1:
        logging.info("Found %d %s operations. Returning the first one.",
                     len(op_list), key)
      if op_list:
        return op_list[0]
    except LookupError:
      pass

    return None

  def _init_ready_op(self,
                     ready_op=USE_DEFAULT,
                     ready_for_local_init_op=USE_DEFAULT):
    """Initializes ready_op.

    Args:
      ready_op: `Tensor` to check if the model is initialized. If it's set to
        USE_DEFAULT, creates an op that checks all the variables are
        initialized.
      ready_for_local_init_op: `Tensor` to check if the model is ready to run
        local_init_op. If it's set to USE_DEFAULT, creates an op that checks all
        the global variables are initialized.
    """
    if ready_op is Supervisor.USE_DEFAULT:
      ready_op = self._get_first_op_from_collection(ops.GraphKeys.READY_OP)
      if ready_op is None:
        ready_op = variables.report_uninitialized_variables()
        ops.add_to_collection(ops.GraphKeys.READY_OP, ready_op)
    self._ready_op = ready_op

    # ready_for_local_init_op defaults to None for backward compatibility
    if ready_for_local_init_op is Supervisor.USE_DEFAULT:
      ready_for_local_init_op = self._get_first_op_from_collection(
          ops.GraphKeys.READY_FOR_LOCAL_INIT_OP)
    self._ready_for_local_init_op = ready_for_local_init_op

  def _init_init_op(self, init_op=USE_DEFAULT, init_feed_dict=None):
    """Initializes init_op.

    Args:
      init_op: `Operation` to initialize the variables. If set to USE_DEFAULT,
        create an op that initializes all variables and tables.
      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
        This feed dictionary will be used when `init_op` is evaluated.
    """
    if init_op is Supervisor.USE_DEFAULT:
      init_op = self._get_first_op_from_collection(ops.GraphKeys.INIT_OP)
      if init_op is None:
        init_op = variables.global_variables_initializer()
        ops.add_to_collection(ops.GraphKeys.INIT_OP, init_op)
    self._init_op = init_op
    self._init_feed_dict = init_feed_dict

  def _init_local_init_op(self, local_init_op=USE_DEFAULT):
    """Initializes local_init_op.

    Args:
      local_init_op: `Operation` run for every new supervisor instance. If set
        to USE_DEFAULT, use the first op from the GraphKeys.LOCAL_INIT_OP
        collection. If the collection is empty, create an op that initializes
        all local variables and all tables.
    """
    if local_init_op is Supervisor.USE_DEFAULT:
      local_init_op = self._get_first_op_from_collection(
          ops.GraphKeys.LOCAL_INIT_OP)
      if local_init_op is None:
        op_list = [
            variables.local_variables_initializer(),
            lookup_ops.tables_initializer()
        ]
        if op_list:
          local_init_op = control_flow_ops.group(*op_list)
          ops.add_to_collection(ops.GraphKeys.LOCAL_INIT_OP, local_init_op)
    self._local_init_op = local_init_op

  def _init_saver(self, saver=USE_DEFAULT):
    """Initializes saver.

    Args:
      saver: A `Saver` object. If set to USE_DEFAULT, create one that saves all
        the variables.
    """
    if saver is Supervisor.USE_DEFAULT:
      saver = self._get_first_op_from_collection(ops.GraphKeys.SAVERS)
      if saver is None and variables.global_variables():
        saver = saver_mod.Saver()
        ops.add_to_collection(ops.GraphKeys.SAVERS, saver)
    self._saver = saver

  def _init_summary_op(self, summary_op=USE_DEFAULT):
    """Initializes summary_op.

    Args:
      summary_op: An Operation that returns a Summary for the event logs. If set
        to USE_DEFAULT, create an op that merges all the summaries.
    """
    if summary_op is Supervisor.USE_DEFAULT:
      summary_op = self._get_first_op_from_collection(ops.GraphKeys.SUMMARY_OP)
      if summary_op is None:
        summary_op = _summary.merge_all()
        if summary_op is not None:
          ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
    self._summary_op = summary_op

  def _init_global_step(self, global_step=USE_DEFAULT):
    """Initializes global_step.

    Args:
      global_step: An integer Tensor of size 1 that counts steps. If set to
        USE_DEFAULT, creates global_step tensor.
    """
    if global_step is Supervisor.USE_DEFAULT:
      global_step = self._get_first_op_from_collection(
          ops.GraphKeys.GLOBAL_STEP)
      if global_step is None:
        global_step = self._default_global_step_tensor()
        if global_step is not None:
          ops.add_to_collection(ops.GraphKeys.GLOBAL_STEP, global_step)
    self._global_step = global_step

  @property
  def is_chief(self):
    """Return True if this is a chief supervisor.

    Returns:
      A bool.
    """
    return self._is_chief

  @property
  def session_manager(self):
    """Return the SessionManager used by the Supervisor.

    Returns:
      A SessionManager object.
    """
    return self._session_manager

  @property
  def coord(self):
    """Return the Coordinator used by the Supervisor.

    The Coordinator can be useful if you want to run multiple threads
    during your training.

    Returns:
      A Coordinator object.
    """
    return self._coord

  @property
  def init_op(self):
    """Return the Init Op used by the supervisor.

    Returns:
      An Op or `None`.
    """
    return self._init_op

  @property
  def init_feed_dict(self):
    """Return the feed dictionary used when evaluating the `init_op`.

    Returns:
      A feed dictionary or `None`.
    """
    return self._init_feed_dict

  @property
  def ready_op(self):
    """Return the Ready Op used by the supervisor.

    Returns:
      An Op or `None`.
    """
    return self._ready_op

  @property
  def ready_for_local_init_op(self):
    return self._ready_for_local_init_op

  @property
  def summary_writer(self):
    """Return the SummaryWriter used by the chief supervisor.

    Returns:
      A SummaryWriter.
    """
    return self._summary_writer

  @property
  def summary_op(self):
    """Return the Summary Tensor used by the chief supervisor.

    Returns:
      A string Tensor for the summary or `None`.
    """
    return self._summary_op

  @property
  def save_summaries_secs(self):
    """Return the delay between summary computations.

    Returns:
      A timestamp.
    """
    return self._save_summaries_secs

  @property
  def global_step(self):
    """Return the global_step Tensor used by the supervisor.

    Returns:
      An integer Tensor for the global_step.
    """
    return self._global_step

  @property
  def saver(self):
    """Return the Saver used by the supervisor.

    Returns:
      A Saver object.
    """
    return self._saver

  @property
  def save_model_secs(self):
    """Return the delay between checkpoints.

    Returns:
      A timestamp.
    """
    return self._save_model_secs

  @property
  def save_path(self):
    """Return the save path used by the supervisor.

    Returns:
      A string.
    """
    return self._save_path

  def _write_graph(self):
    """Writes graph_def to `logdir` and adds it to summary if applicable."""
    assert self._is_chief
    if self._logdir:
      training_util.write_graph(
          self._graph.as_graph_def(add_shapes=True), self._logdir,
          "graph.pbtxt")
    if self._summary_writer and not self._graph_added_to_summary:
      self._summary_writer.add_graph(self._graph)
      self._summary_writer.add_meta_graph(self._meta_graph_def)
      self._graph_added_to_summary = True

  def start_standard_services(self, sess):
    """Start the standard services for 'sess'.

    This starts services in the background.  The services started depend
    on the parameters to the constructor and may include:

      - A Summary thread computing summaries every save_summaries_secs.
      - A Checkpoint thread saving the model every save_model_secs.
      - A StepCounter thread measure step time.

    Args:
      sess: A Session.

    Returns:
      A list of threads that are running the standard services.  You can use
      the Supervisor's Coordinator to join these threads with:
        sv.coord.Join(<list of threads>)

    Raises:
      RuntimeError: If called with a non-chief Supervisor.
      ValueError: If not `logdir` was passed to the constructor as the
        services need a log directory.
    """
    if not self._is_chief:
      raise RuntimeError("Only chief supervisor can start standard services. "
                         "Because only chief supervisors can write events.")

    if not self._logdir:
      logging.warning("Standard services need a 'logdir' "
                      "passed to the SessionManager")
      return

    if self._global_step is not None and self._summary_writer:
      # Only add the session log if we keep track of global step.
      # TensorBoard cannot use START message for purging expired events
      # if there is no step value.
      current_step = training_util.global_step(sess, self._global_step)
      self._summary_writer.add_session_log(
          SessionLog(status=SessionLog.START), current_step)

    threads = []
    if self._save_summaries_secs and self._summary_writer:
      if self._summary_op is not None:
        threads.append(SVSummaryThread(self, sess))
      if self._global_step is not None:
        threads.append(SVStepCounterThread(self, sess))
    if self.saver and self._save_model_secs:
      threads.append(SVTimerCheckpointThread(self, sess))
    for t in threads:
      t.start()
    return threads

  def prepare_or_wait_for_session(self,
                                  master="",
                                  config=None,
                                  wait_for_checkpoint=False,
                                  max_wait_secs=7200,
                                  start_standard_services=True):
    """Make sure the model is ready to be used.

    Create a session on 'master', recovering or initializing the model as
    needed, or wait for a session to be ready.  If running as the chief
    and `start_standard_service` is set to True, also call the session
    manager to start the standard services.

    Args:
      master: name of the TensorFlow master to use.  See the
        `tf.compat.v1.Session` constructor for how this is interpreted.
      config: Optional ConfigProto proto used to configure the session, which is
        passed as-is to create the session.
      wait_for_checkpoint: Whether we should wait for the availability of a
        checkpoint before creating Session. Defaults to False.
      max_wait_secs: Maximum time to wait for the session to become available.
      start_standard_services: Whether to start the standard services and the
        queue runners.

    Returns:
      A Session object that can be used to drive the model.
    """
    # For users who recreate the session with prepare_or_wait_for_session(), we
    # need to clear the coordinator's stop_event so that threads managed by the
    # coordinator can run.
    self._coord.clear_stop()
    if self._summary_writer:
      self._summary_writer.reopen()

    if self._is_chief:
      sess = self._session_manager.prepare_session(
          master,
          init_op=self.init_op,
          saver=self.saver,
          checkpoint_dir=self._logdir,
          wait_for_checkpoint=wait_for_checkpoint,
          max_wait_secs=max_wait_secs,
          config=config,
          init_feed_dict=self._init_feed_dict,
          init_fn=self._init_fn)
      self._write_graph()
      if start_standard_services:
        logging.info("Starting standard services.")
        self.start_standard_services(sess)
    else:
      sess = self._session_manager.wait_for_session(
          master, config=config, max_wait_secs=max_wait_secs)
    if start_standard_services:
      logging.info("Starting queue runners.")
      self.start_queue_runners(sess)
    return sess

  def start_queue_runners(self, sess, queue_runners=None):
    """Start threads for `QueueRunners`.

    Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
    are already started automatically when you create a session with the
    supervisor, so unless you have non-collected queue runners to start
    you do not need to call this explicitly.

    Args:
      sess: A `Session`.
      queue_runners: A list of `QueueRunners`. If not specified, we'll use the
        list of queue runners gathered in the graph under the key
        `GraphKeys.QUEUE_RUNNERS`.

    Returns:
      The list of threads started for the `QueueRunners`.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
    Queues are not compatible with eager execution. To ingest data when eager
    execution is enabled, use the `tf.data` API.
    @end_compatibility
    """
    if context.executing_eagerly():
      raise RuntimeError("Queues are not compatible with eager execution.")
    if queue_runners is None:
      queue_runners = self._graph.get_collection(ops.GraphKeys.QUEUE_RUNNERS)
    threads = []
    for qr in queue_runners:
      threads.extend(
          qr.create_threads(sess, coord=self._coord, daemon=True, start=True))
    return threads

  def loop(self, timer_interval_secs, target, args=None, kwargs=None):
    """Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
    repeatedly.  Otherwise it calls it every `timer_interval_secs`
    seconds.  The thread terminates when a stop is requested.

    The started thread is added to the list of threads managed by the supervisor
    so it does not need to be passed to the `stop()` method.

    Args:
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    """
    looper = coordinator.LooperThread(
        self._coord,
        timer_interval_secs,
        target=target,
        args=args,
        kwargs=kwargs)
    looper.start()
    return looper

  def stop(self,
           threads=None,
           close_summary_writer=True,
           ignore_live_threads=False):
    """Stop the services and the coordinator.

    This does not close the session.

    Args:
      threads: Optional list of threads to join with the coordinator.  If
        `None`, defaults to the threads running the standard services, the
        threads started for `QueueRunners`, and the threads started by the
        `loop()` method.  To wait on additional threads, pass the list in this
        parameter.
      close_summary_writer: Whether to close the `summary_writer`.  Defaults to
        `True` if the summary writer was created by the supervisor, `False`
        otherwise.
      ignore_live_threads: If `True` ignores threads that remain running after a
        grace period when joining threads via the coordinator, instead of
        raising a RuntimeError.
    """
    self._coord.request_stop()
    try:
      # coord.join() re-raises the first reported exception; the "finally"
      # block ensures that we clean up whether or not an exception was
      # reported.
      self._coord.join(
          threads,
          stop_grace_period_secs=self._stop_grace_secs,
          ignore_live_threads=ignore_live_threads)
    finally:
      # Close the writer last, in case one of the running threads was using it.
      if close_summary_writer and self._summary_writer:
        # Stop messages are not logged with event.step,
        # since the session may have already terminated.
        self._summary_writer.add_session_log(SessionLog(status=SessionLog.STOP))
        self._summary_writer.close()
        self._graph_added_to_summary = False

  def request_stop(self, ex=None):
    """Request that the coordinator stop the threads.

    See `Coordinator.request_stop()`.

    Args:
      ex: Optional `Exception`, or Python `exc_info` tuple as returned by
        `sys.exc_info()`.  If this is the first call to `request_stop()` the
        corresponding exception is recorded and re-raised from `join()`.
    """
    self._coord.request_stop(ex=ex)

  def should_stop(self):
    """Check if the coordinator was told to stop.

    See `Coordinator.should_stop()`.

    Returns:
      True if the coordinator was told to stop, False otherwise.
    """
    return self._coord.should_stop()

  def stop_on_exception(self):
    """Context handler to stop the supervisor when an exception is raised.

    See `Coordinator.stop_on_exception()`.

    Returns:
      A context handler.
    """
    return self._coord.stop_on_exception()

  def wait_for_stop(self):
    """Block waiting for the coordinator to stop."""
    self._coord.wait_for_stop()

  def summary_computed(self, sess, summary, global_step=None):
    """Indicate that a summary was computed.

    Args:
      sess: A `Session` object.
      summary: A Summary proto, or a string holding a serialized summary proto.
      global_step: Int. global step this summary is associated with. If `None`,
        it will try to fetch the current step.

    Raises:
      TypeError: if 'summary' is not a Summary proto or a string.
      RuntimeError: if the Supervisor was created without a `logdir`.
    """
    if not self._summary_writer:
      raise RuntimeError("Writing a summary requires a summary writer.")
    if global_step is None and self.global_step is not None:
      global_step = training_util.global_step(sess, self.global_step)
    self._summary_writer.add_summary(summary, global_step)

  def _default_global_step_tensor(self):
    """Returns the global_step from the default graph.

    Returns:
      The global step `Tensor` or `None`.
    """
    try:
      gs = ops.get_default_graph().get_tensor_by_name("global_step:0")
      if gs.dtype.base_dtype in [dtypes.int32, dtypes.int64]:
        return gs
      else:
        logging.warning("Found 'global_step' is not an int type: %s", gs.dtype)
        return None
    except KeyError:
      return None

  def _verify_setup(self):
    """Check that all is good.

    Raises:
      ValueError: If something is not good.
    """
    # Not running as chief means that replicas are used.
    # In that case all Variables must have their device set.
    if not self._is_chief:
      for op in self._graph.get_operations():
        if op.type in ["Variable", "VariableV2"] and not op.device:
          raise ValueError("When using replicas, all Variables must have "
                           "their device set: %s" % op)

  # pylint: disable=g-doc-return-or-yield,broad-except
  @contextlib.contextmanager
  def managed_session(self,
                      master="",
                      config=None,
                      start_standard_services=True,
                      close_summary_writer=True):
    """Returns a context manager for a managed session.

    This context manager creates and automatically recovers a session.  It
    optionally starts the standard services that handle checkpoints and
    summaries.  It monitors exceptions raised from the `with` block or from the
    services and stops the supervisor as needed.

    The context manager is typically used as follows:

    ```python
    def train():
      sv = tf.compat.v1.train.Supervisor(...)
      with sv.managed_session(<master>) as sess:
        for step in xrange(..):
          if sv.should_stop():
            break
          sess.run(<my training op>)
          ...do other things needed at each training step...
    ```

    An exception raised from the `with` block or one of the service threads is
    raised again when the block exits.  This is done after stopping all threads
    and closing the session.  For example, an `AbortedError` exception, raised
    in case of preemption of one of the workers in a distributed model, is
    raised again when the block exits.

    If you want to retry the training loop in case of preemption you can do it
    as follows:

    ```python
    def main(...):
      while True
        try:
          train()
        except tf.errors.Aborted:
          pass
    ```

    As a special case, exceptions used for control flow, such as
    `OutOfRangeError` which reports that input queues are exhausted, are not
    raised again from the `with` block: they indicate a clean termination of
    the training loop and are considered normal termination.

    Args:
      master: name of the TensorFlow master to use.  See the
        `tf.compat.v1.Session` constructor for how this is interpreted.
      config: Optional `ConfigProto` proto used to configure the session. Passed
        as-is to create the session.
      start_standard_services: Whether to start the standard services, such as
        checkpoint, summary and step counter.
      close_summary_writer: Whether to close the summary writer when closing the
        session.  Defaults to True.

    Returns:
      A context manager that yields a `Session` restored from the latest
      checkpoint or initialized from scratch if not checkpoint exists.  The
      session is closed when the `with` block exits.
    """
    try:
      sess = self.prepare_or_wait_for_session(
          master=master,
          config=config,
          start_standard_services=start_standard_services)
      yield sess
    except Exception as e:
      self.request_stop(e)
    finally:
      try:
        # Request all the threads to stop and wait for them to do so.  Any
        # exception raised by the threads is raised again from stop().
        # Passing stop_grace_period_secs is for blocked enqueue/dequeue
        # threads which are not checking for `should_stop()`.  They
        # will be stopped when we close the session further down.
        self.stop(close_summary_writer=close_summary_writer)
      finally:
        # Close the session to finish up all pending calls.  We do not care
        # about exceptions raised when closing.  This takes care of
        # blocked enqueue/dequeue calls.
        try:
          sess.close()
        except Exception:
          # Silently ignore exceptions raised by close().
          pass

  # pylint: enable=g-doc-return-or-yield,broad-except


class SVSummaryThread(coordinator.LooperThread):
  """A thread to save summaries on a timer."""

  def __init__(self, sv, sess):
    """Create a SVSummaryThread.

    Args:
      sv: A `Supervisor`.
      sess: A `Session`.
    """
    super(SVSummaryThread, self).__init__(sv.coord, sv.save_summaries_secs)
    self._sv = sv
    self._sess = sess

  def run_loop(self):
    if self._sv.global_step is not None:
      summary_strs, global_step = self._sess.run(
          [self._sv.summary_op, self._sv.global_step])
    else:
      summary_strs = self._sess.run(self._sv.summary_op)
      global_step = None
    if self._sv.summary_writer:
      logging.info("Recording summary at step %s.", global_step)
      self._sv.summary_writer.add_summary(summary_strs, global_step)


class SVStepCounterThread(coordinator.LooperThread):
  """Threads to count steps and measure their duration."""

  def __init__(self, sv, sess, step_counter=None):
    """Create a `SVStepCounterThread`.

    Args:
      sv: A `Supervisor`.
      sess: A `Session`.
      step_counter: A `Tensor` holding the step counter. By defaults, it uses
        sv.global_step.
    """
    super(SVStepCounterThread, self).__init__(sv.coord, sv.save_summaries_secs)
    self._sv = sv
    self._sess = sess
    self._last_time = 0.0
    self._last_step = 0
    step_counter = sv.global_step if step_counter is None else step_counter
    self._step_counter = step_counter
    self._summary_tag = "%s/sec" % self._step_counter.op.name

  def start_loop(self):
    self._last_time = time.time()
    self._last_step = training_util.global_step(self._sess, self._step_counter)

  def run_loop(self):
    # Count the steps.
    current_step = training_util.global_step(self._sess, self._step_counter)
    added_steps = current_step - self._last_step
    self._last_step = current_step
    # Measure the elapsed time.
    current_time = time.time()
    elapsed_time = current_time - self._last_time
    self._last_time = current_time
    # Reports the number of steps done per second
    if elapsed_time > 0.:
      steps_per_sec = added_steps / elapsed_time
    else:
      steps_per_sec = float("inf")
    summary = Summary(value=[
        Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)
    ])
    if self._sv.summary_writer:
      self._sv.summary_writer.add_summary(summary, current_step)
    logging.log_first_n(logging.INFO, "%s: %g", 10, self._summary_tag,
                        steps_per_sec)


class SVTimerCheckpointThread(coordinator.LooperThread):
  """A thread to checkpoint on a timer."""

  def __init__(self, sv, sess):
    """Create a `SVTimerCheckpointThread`.

    Args:
      sv: A `Supervisor`.
      sess: A `Session`.
    """
    super(SVTimerCheckpointThread, self).__init__(sv.coord, sv.save_model_secs)
    self._sv = sv
    self._sess = sess

  def run_loop(self):
    logging.info("Saving checkpoint to path %s", self._sv.save_path)
    self._sv.saver.save(
        self._sess, self._sv.save_path, global_step=self._sv.global_step)
    if self._sv.summary_writer and self._sv.global_step is not None:
      current_step = training_util.global_step(self._sess, self._sv.global_step)
      self._sv.summary_writer.add_session_log(
          SessionLog(
              status=SessionLog.CHECKPOINT, checkpoint_path=self._sv.save_path),
          current_step)


# TODO(sherrym): All non-PEP8 compliant names will be deprecated shortly.
setattr(Supervisor, "PrepareSession", Supervisor.prepare_or_wait_for_session)
setattr(Supervisor, "StartQueueRunners", Supervisor.start_queue_runners)
setattr(Supervisor, "StartStandardServices", Supervisor.start_standard_services)
setattr(Supervisor, "Stop", Supervisor.stop)
setattr(Supervisor, "RequestStop", Supervisor.request_stop)
setattr(Supervisor, "Loop", Supervisor.loop)
setattr(Supervisor, "ShouldStop", Supervisor.should_stop)
setattr(Supervisor, "StopOnException", Supervisor.stop_on_exception)
setattr(Supervisor, "WaitForStop", Supervisor.wait_for_stop)
setattr(Supervisor, "SummaryComputed", Supervisor.summary_computed)
