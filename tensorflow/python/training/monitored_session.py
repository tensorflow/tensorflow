# pylint: disable=g-bad-file-header
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
"""A wrapper of Session API which runs hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook


# TODO(touts): Share that with the Supervisor.
class Scaffold(object):
  """Structure to create or gather pieces commonly needed to train a model.

  When you build a model for training you usually need ops to initialize
  variables, a `Saver` to checkpoint them, an op to collect summaries for
  the visualizer, and so on.

  Various libraries built on top of the core TensorFlow library take care of
  creating some or all of these pieces and storing them in well known
  collections in the graph.  The `Scaffold` class helps pick these pieces from
  the graph collections, creating and adding them to the collections if needed.

  If you call the scaffold constructor without any arguments, it will pick
  pieces from the collections, creating default ones if needed when
  `scaffold.finalize()` is called.  You can pass arguments to the constructor to
  provide your own pieces.  Pieces that you pass to the constructor are not
  added to the graph collections.

  The following pieces are directly accessible as attributes of the `Scaffold`
  object:

  * `saver`: A `tf.Saver` object taking care of saving the variables.  Picked
    from and stored into the `SAVERS` collection in the graph by default.
  * `init_op`: An op to run to initialize the variables.  Picked from and
    stored into the `INIT_OP` collection in the graph by default.
  * `ready_op`: An op to verify that the variables are initialized.  Picked
    from and stored into the `READY_OP` collection in the graph by default.
  * `ready_for_local_init_op`: An op to verify that global state has been
    initialized and it is alright to run `local_init_op`.  Picked from and
    stored into the `READY_FOR_LOCAL_INIT_OP` collection in the graph by
    default. This is needed when the initialization of local variables depends
    on the values of global variables.
  * `local_init_op`: An op to initialize the local variables.  Picked
    from and stored into the `LOCAL_INIT_OP` collection in the graph by default.
  * `summary_op`: An op to run and merge the summaries in the graph.  Picked
    from and stored into the `SUMMARY_OP` collection in the graph by default.
  * `global_step`: A tensor containing the global step counter.  Picked
    from and stored into the `GLOBAL_STEP` collection in the graph by default.

  You can also pass the following additional pieces to the constructor:

  * `init_feed_dict`: A sessionn feed dictionary that should be used when
     running the init op.
  * `init_fn`: A callable to run run after the init op to perform additional
    initializations.  The callable will be called as
    `init_fn(scaffold, session)`.

  """

  def __init__(self,
               init_op=None,
               init_feed_dict=None,
               init_fn=None,
               ready_op=None,
               ready_for_local_init_op=None,
               local_init_op=None,
               summary_op=None,
               saver=None):
    """Create a scaffold.

    Args:
      init_op: Optional op for initializing variables.
      init_feed_dict: Optional session feed dictionary to use when running the
        init_op.
      init_fn: Optional function to use to initialize the model after running
        the init_op.  Will be called as `init_fn(scaffold, session)`.
      ready_op: Optional op to verify that the variables are initialized.  Must
        return an empty 1D string tensor when the variables are initialized, or
        a non-empty 1D string tensor listing the names of the non-initialized
        variables.
      ready_for_local_init_op: Optional op to verify that the global variables
        are initialized and `local_init_op` can be run. Must return an empty
        1D string tensor when the global variables are initialized, or a
        non-empty 1D string tensor listing the names of the non-initialized
        global variables.
      local_init_op: Optional op to initialize local variables.
      summary_op: Optional op to gather all summaries.  Must return a scalar
        string tensor containing a serialized `Summary` proto.
      saver: Optional `tf.Saver` object to use to save and restore variables.
    """

    # NOTE(touts): modifying the init function to be passed the scaffold is a
    # hack to make it easy to find the saver.  Is there a better way?
    if init_fn:
      self._init_fn = lambda sess: init_fn(self, sess)
    else:
      self._init_fn = None

    self._init_op = init_op
    self._ready_op = ready_op
    self._ready_for_local_init_op = ready_for_local_init_op
    self._local_init_op = local_init_op
    self._summary_op = summary_op
    self._saver = saver
    self._init_feed_dict = init_feed_dict

  def finalize(self):
    """Creates operations if needed and finalizes the graph."""
    if self._init_op is None:
      def default_init_op():
        return control_flow_ops.group(
            variables.global_variables_initializer(),
            resources.initialize_resources(resources.shared_resources()))
      self._init_op = Scaffold.get_or_default(
          'init_op',
          ops.GraphKeys.INIT_OP,
          default_init_op)
    if self._ready_op is None:
      def default_ready_op():
        return array_ops.concat([
            variables.report_uninitialized_variables(),
            resources.report_uninitialized_resources()
        ], 0)
      self._ready_op = Scaffold.get_or_default(
          'ready_op', ops.GraphKeys.READY_OP,
          default_ready_op)
    if self._ready_for_local_init_op is None:
      def default_ready_for_local_init_op():
        return variables.report_uninitialized_variables(
            variables.global_variables())
      self._ready_for_local_init_op = Scaffold.get_or_default(
          'ready_for_local_init_op', ops.GraphKeys.READY_FOR_LOCAL_INIT_OP,
          default_ready_for_local_init_op)
    if self._local_init_op is None:
      self._local_init_op = Scaffold.get_or_default(
          'local_init_op', ops.GraphKeys.LOCAL_INIT_OP,
          Scaffold._default_local_init_op)
    if self._summary_op is None:
      self._summary_op = Scaffold.get_or_default('summary_op',
                                                 ops.GraphKeys.SUMMARY_OP,
                                                 summary.merge_all)
    # pylint: disable=g-long-lambda
    if self._saver is None:
      self._saver = Scaffold.get_or_default(
          'saver',
          ops.GraphKeys.SAVERS,
          lambda: training_saver.Saver(sharded=True, allow_empty=True,
                                       write_version=saver_pb2.SaverDef.V2))
    # pylint: enable=g-long-lambda
    self._saver.build()

    ops.get_default_graph().finalize()
    return self

  @property
  def init_fn(self):
    return self._init_fn

  @property
  def init_op(self):
    return self._init_op

  @property
  def ready_op(self):
    return self._ready_op

  @property
  def ready_for_local_init_op(self):
    return self._ready_for_local_init_op

  @property
  def local_init_op(self):
    return self._local_init_op

  @property
  def summary_op(self):
    return self._summary_op

  @property
  def saver(self):
    return self._saver

  @property
  def init_feed_dict(self):
    return self._init_feed_dict

  @staticmethod
  def get_or_default(arg_name, collection_key, default_constructor):
    """Get from cache or create a default operation."""
    elements = ops.get_collection(collection_key)
    if elements:
      if len(elements) > 1:
        raise RuntimeError('More than one item in the collection "%s". '
                           'Please indicate which one to use by passing it to '
                           'the tf.Scaffold constructor as:  '
                           'tf.Scaffold(%s=item to use)', collection_key,
                           arg_name)
      return elements[0]
    op = default_constructor()
    if op is not None:
      ops.add_to_collection(collection_key, op)
    return op

  @staticmethod
  def _default_local_init_op():
    return control_flow_ops.group(variables.local_variables_initializer(),
                                  data_flow_ops.tables_initializer())


def MonitoredTrainingSession(master='',  # pylint: disable=invalid-name
                             is_chief=True,
                             checkpoint_dir=None,
                             scaffold=None,
                             hooks=None,
                             chief_only_hooks=None,
                             save_checkpoint_secs=600,
                             save_summaries_steps=100,
                             save_summaries_secs=None,
                             config=None,
                             stop_grace_period_secs=120):
  """Creates a `MonitoredSession` for training.

  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  inialize/restore.


  Args:
    master: `String` the TensorFlow master to use.
    is_chief: If `True`, it will take care of initialization and recovery the
      underlying TensorFlow session. If `False`, it will wait on a chief to
      initialize or recover the TensorFlow session.
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If
      not specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used.
    config: an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.

  Returns:
    A `MonitoredSession` object.
  """
  scaffold = scaffold or Scaffold()
  if not is_chief:
    session_creator = WorkerSessionCreator(
        scaffold=scaffold, master=master, config=config)
    return MonitoredSession(session_creator=session_creator, hooks=hooks or [],
                            stop_grace_period_secs=stop_grace_period_secs)

  all_hooks = []
  if chief_only_hooks:
    all_hooks.extend(chief_only_hooks)
  session_creator = ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_dir=checkpoint_dir,
      master=master,
      config=config)

  if checkpoint_dir:
    all_hooks.append(
        basic_session_run_hooks.StepCounterHook(output_dir=checkpoint_dir))

    if (save_summaries_steps and save_summaries_steps > 0) or (
        save_summaries_secs and save_summaries_secs > 0):
      all_hooks.append(basic_session_run_hooks.SummarySaverHook(
          scaffold=scaffold,
          save_steps=save_summaries_steps,
          save_secs=save_summaries_secs,
          output_dir=checkpoint_dir))
    if save_checkpoint_secs and save_checkpoint_secs > 0:
      all_hooks.append(basic_session_run_hooks.CheckpointSaverHook(
          checkpoint_dir, save_secs=save_checkpoint_secs, scaffold=scaffold))

  if hooks:
    all_hooks.extend(hooks)
  return MonitoredSession(session_creator=session_creator, hooks=all_hooks,
                          stop_grace_period_secs=stop_grace_period_secs)


class SessionCreator(object):
  """A factory for tf.Session."""

  @abc.abstractmethod
  def create_session(self):
    raise NotImplementedError(
        'create_session is not implemented for {}.'.format(self))


class ChiefSessionCreator(SessionCreator):
  """Creates a tf.Session  for a chief."""

  def __init__(self,
               scaffold=None,
               master='',
               config=None,
               checkpoint_dir=None,
               checkpoint_filename_with_path=None):
    """Initializes a chief session creator.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string.  Optional path to a directory where to restore
        variables.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
    """
    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_filename_with_path = checkpoint_filename_with_path
    self._scaffold = scaffold or Scaffold()
    self._session_manager = None
    self._master = master
    self._config = config

  def _get_session_manager(self):
    if self._session_manager:
      return self._session_manager

    self._session_manager = sm.SessionManager(
        local_init_op=self._scaffold.local_init_op,
        ready_op=self._scaffold.ready_op,
        ready_for_local_init_op=self._scaffold.ready_for_local_init_op,
        graph=ops.get_default_graph())
    return self._session_manager

  def create_session(self):
    self._scaffold.finalize()
    return self._get_session_manager().prepare_session(
        self._master,
        saver=self._scaffold.saver,
        checkpoint_dir=self._checkpoint_dir,
        checkpoint_filename_with_path=self._checkpoint_filename_with_path,
        config=self._config,
        init_op=self._scaffold.init_op,
        init_feed_dict=self._scaffold.init_feed_dict,
        init_fn=self._scaffold.init_fn)


class WorkerSessionCreator(SessionCreator):
  """Creates a tf.Session for a worker."""

  def __init__(self, scaffold=None, master='', config=None):
    """Initializes a worker session creator.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
    """
    self._scaffold = scaffold or Scaffold()
    self._session_manager = None
    self._master = master
    self._config = config

  def _get_session_manager(self):
    if self._session_manager:
      return self._session_manager

    self._session_manager = sm.SessionManager(
        local_init_op=self._scaffold.local_init_op,
        ready_op=self._scaffold.ready_op,
        ready_for_local_init_op=self._scaffold.ready_for_local_init_op,
        graph=ops.get_default_graph())
    return self._session_manager

  def create_session(self):
    self._scaffold.finalize()
    return self._get_session_manager().wait_for_session(
        self._master, config=self._config)


class _MonitoredSession(object):
  """See `MonitoredSession` or `SingularMonitoredSession`."""

  def __init__(self, session_creator, hooks, should_recover,
               stop_grace_period_secs=120):
    """Sets up a Monitored or Hooked Session.

    Args:
      session_creator: A factory object to create session. Typically a
        `ChiefSessionCreator` or a `WorkerSessionCreator`.
      hooks: An iterable of `SessionRunHook' objects.
      should_recover: A bool. Indicates whether to recover from `AbortedError`
        or not.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
    """
    self._graph_was_finalized = ops.get_default_graph().finalized
    self._hooks = hooks or []
    for h in self._hooks:
      h.begin()
    # Create the session.
    self._coordinated_creator = self._CoordinatedSessionCreator(
        session_creator=session_creator or ChiefSessionCreator(),
        hooks=self._hooks,
        stop_grace_period_secs=stop_grace_period_secs)
    if should_recover:
      self._sess = _RecoverableSession(self._coordinated_creator)
    else:
      self._sess = self._coordinated_creator.create_session()

  @property
  def graph(self):
    """The graph that was launched in this session."""
    if self._tf_sess() is None:
      return None
    return self._tf_sess().graph

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Run ops in the monitored session.

    This method is completely compatible with the `tf.Session.run()` method.

    Args:
      fetches: Same as `tf.Session.run()`.
      feed_dict: Same as `tf.Session.run()`.
      options: Same as `tf.Session.run()`.
      run_metadata: Same as `tf.Session.run()`.

    Returns:
      Same as `tf.Session.run()`.
    """
    return self._sess.run(fetches,
                          feed_dict=feed_dict,
                          options=options,
                          run_metadata=run_metadata)

  def should_stop(self):
    if self._sess:
      return self._sess.should_stop()
    return True

  def close(self):
    self._close_internal()

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    if exception_type in [errors.OutOfRangeError, StopIteration]:
      exception_type = None
    self._close_internal(exception_type)
    # __exit__ should return True to suppress an exception.
    return exception_type is None

  class _CoordinatedSessionCreator(object):
    """Factory for the _RecoverableSession."""

    def __init__(self, session_creator, hooks, stop_grace_period_secs):
      self._session_creator = session_creator
      self._hooks = hooks
      self.coord = None
      self.tf_sess = None
      self._stop_grace_period_secs = stop_grace_period_secs

    def create_session(self):
      """Creates a coordinated session."""
      # Keep the tf_sess for unit testing.
      self.tf_sess = self._session_creator.create_session()
      # We don't want coordinator to suppress any exception.
      self.coord = coordinator.Coordinator(clean_stop_exception_types=[])
      queue_runner.start_queue_runners(sess=self.tf_sess, coord=self.coord)
      # Inform the hooks that a new session has been created.
      for hook in self._hooks:
        hook.after_create_session(self.tf_sess, self.coord)
      return _CoordinatedSession(
          _HookedSession(self.tf_sess, self._hooks), self.coord,
          self._stop_grace_period_secs)

  def _close_internal(self, exception_type=None):
    try:
      if not exception_type:
        for h in self._hooks:
          h.end(self._coordinated_creator.tf_sess)
    finally:
      try:
        self._sess.close()
      finally:
        self._sess = None
        self._coordinated_creator.tf_sess = None
        self._coordinated_creator.coord = None
        if not self._graph_was_finalized:
          ops.get_default_graph()._unsafe_unfinalize()  # pylint: disable=protected-access

  def _is_closed(self):
    """Return True if the supervised session is closed.  For tests only.

    Returns:
      A boolean.
    """
    return self._coordinated_creator.tf_sess is None

  def _tf_sess(self):
    return self._coordinated_creator.tf_sess


class MonitoredSession(_MonitoredSession):
  """Session-like object that handles initialization, recovery and hooks.

  Example usage:

  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummaryHook(...)
  with MonitoredSession(session_creator=ChiefSessionCreator(...),
                        hooks=[saver_hook, summary_hook]) as sess:
    while not sess.should_stop():
      sess.run(train_op)
  ```

  Initialization: At creation time the monitored session does following things
  in given order:

  * calls `hook.begin()` for each given hook
  * finalizes the graph via `scaffold.finalize()`
  * create session
  * initializes the model via initialization ops provided by `Scaffold`
  * restores variables if a checkpoint exists
  * launches queue runners

  Run: When `run()` is called, the monitored session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user
  * if `AbortedError` occurs, it recovers or reinitializes the session before
    executing the run() call again


  Exit: At the `close()`, the monitored session does following things in order:

  * calls `hook.end()`
  * closes the queue runners and the session
  * suppresses `OutOfRange` error which indicates that all inputs have been
    processed if the monitored_session is used as a context

  How to set `tf.Session` arguments:

  * In most cases you can set session arguments as follows:

  ```python
  MonitoredSession(
    session_creator=ChiefSessionCreator(master=..., config=...))
  ```

  * In distributed setting for a non-chief worker, you can use following:

  ```python
  MonitoredSession(
    session_creator=WorkerSessionCreator(master=..., config=...))
  ```

  See `MonitoredTrainingSession` for an example usage based on chief or worker.

  Args:
    session_creator: A factory object to create session. Typically a
      `ChiefSessionCreator` which is the default one.
    hooks: An iterable of `SessionRunHook' objects.

  Returns:
    A MonitoredSession object.
  """

  def __init__(self, session_creator=None, hooks=None,
               stop_grace_period_secs=120):
    super(MonitoredSession, self).__init__(
        session_creator, hooks, should_recover=True,
        stop_grace_period_secs=stop_grace_period_secs)


class SingularMonitoredSession(_MonitoredSession):
  """Session-like object that handles initialization, restoring, and hooks.

  Please note that this utility is not recommended for distributed settings.
  For distributed settings, please use `tf.train.MonitoredSession`. The
  differences between `MonitoredSession` and `SingularMonitoredSession` are:
  * `MonitoredSession` handles `AbortedError` for distributed settings,
    but `SingularMonitoredSession` does not.
  * `MonitoredSession` can be created in `chief` or `worker` modes.
    `SingularMonitoredSession` is always created as `chief`.
  * You can access the raw `tf.Session` object used by
    `SingularMonitoredSession`, whereas in MonitoredSession the raw session is
    private. This can be used:
    - To `run` without hooks.
    - To save and restore.
  * All other functionality is identical.

  Example usage:
  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummaryHook(...)
  with SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
    while not sess.should_stop():
      sess.run(train_op)
  ```

  Initialization: At creation time the hooked session does following things
  in given order:

  * calls `hook.begin()` for each given hook
  * finalizes the graph via `scaffold.finalize()`
  * create session
  * initializes the model via initialization ops provided by `Scaffold`
  * restores variables if a checkpoint exists
  * launches queue runners

  Run: When `run()` is called, the hooked session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user

  Exit: At the `close()`, the hooked session does following things in order:

  * calls `hook.end()`
  * closes the queue runners and the session
  * surpresses `OutOfRange` error which indicates that all inputs have been
    processed if the `SingularMonitoredSession` is used as a context.
  """

  def __init__(self,
               hooks=None,
               scaffold=None,
               master='',
               config=None,
               checkpoint_dir=None,
               stop_grace_period_secs=120):
    """Creates a SingularMonitoredSession.

    Args:
      hooks: An iterable of `SessionRunHook' objects.
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string.  Optional path to a directory where to restore
        variables.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
    """
    session_creator = ChiefSessionCreator(
        scaffold=scaffold,
        master=master,
        config=config,
        checkpoint_dir=checkpoint_dir)
    super(SingularMonitoredSession, self).__init__(
        session_creator, hooks, should_recover=False,
        stop_grace_period_secs=stop_grace_period_secs)

  def raw_session(self):
    """Returns underlying `TensorFlow.Session` object."""
    return self._tf_sess()


class _WrappedSession(object):
  """Wrapper around a `tf.Session`.

  This wrapper is used as a base class for various session wrappers
  that provide additional functionality such as monitoring, coordination,
  and recovery.

  In addition to the methods exported by `SessionInterface` the wrapper
  provides a method to check for stop and never raises exceptions from
  calls to `close()`.
  """

  def __init__(self, sess):
    """Creates a `_WrappedSession`.

    Args:
      sess: A `tf.Session` or `_WrappedSession` object.  The wrapped session.
    """
    self._sess = sess
    self._wrapped_is_stoppable = isinstance(self._sess, _WrappedSession)

  @property
  def graph(self):
    return self._sess.graph

  @property
  def sess_str(self):
    return self._sess.sess_str

  def should_stop(self):
    """Return true if this session should not be used anymore.

    Always return True if the session was closed.

    Returns:
      True if the session should stop, False otherwise.
    """
    if self._check_stop():
      return True
    if self._sess:
      return self._wrapped_is_stoppable and self._sess.should_stop()
    return True

  def _check_stop(self):
    """Hook for subclasses to provide their own stop condition.

    Returns:
      True if the session should stop, False otherwise.
    """
    return False

  def close(self):
    if self._sess:
      try:
        self._sess.close()
      finally:
        self._sess = None

  def run(self, *args, **kwargs):
    return self._sess.run(*args, **kwargs)


class _RecoverableSession(_WrappedSession):
  """A wrapped session that recreates a session on `tf.errors.AbortedError`.

  The constructor is passed a SessionCreator object, not a session.

  Calls to `run()` are delegated to the wrapped session.  If a call raises the
  exception `tf.errors.AbortedError`, the wrapped session is closed, and a new
  one is created by calling the factory again.
  """

  def __init__(self, sess_creator):
    """Create a new `_RecoverableSession`.

    The value returned by calling `sess_creator.create_session()` will be the
    session wrapped by this recoverable session.

    Args:
      sess_creator: A 'SessionCreator' to be wrapped by recoverable.
    """
    self._sess_creator = sess_creator
    _WrappedSession.__init__(self, self._create_session())

  def _create_session(self):
    while True:
      try:
        return self._sess_creator.create_session()
      except errors.AbortedError:
        logging.info('An AbortedError was raised during initialization. '
                     'It\'s most likely due to a preemption in a connected '
                     'worker/ps. A new session will be created.')

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    while True:
      try:
        if not self._sess:
          self._sess = self._create_session()
        return self._sess.run(fetches,
                              feed_dict=feed_dict,
                              options=options,
                              run_metadata=run_metadata)
      except errors.AbortedError:
        logging.info('An AbortedError was raised. Closing the current session. '
                     'It\'s most likely due to a preemption in a connected '
                     'worker/ps. '
                     'A new session will be created on the next session.run().')
        self.close()
        self._sess = None


class _CoordinatedSession(_WrappedSession):
  """A wrapped session that works with a `tf.Coordinator`.

  Calls to `run()` are delegated to the wrapped session.  If a call
  raises an exception, the exception is reported to the coordinator.

  In addition, after each call to `run()` this session ask the coordinator if
  the session should stop.  In that case it will will join all the threads
  registered with the coordinator before returning.

  If the coordinator was requested to stop with an exception, that exception
  will be re-raised from the call to `run()`.
  """

  def __init__(self, sess, coord, stop_grace_period_secs=120):
    """Create a new `_CoordinatedSession`.

    Args:
      sess: A `tf.Session` object.  The wrapped session.
      coord: A `tf.train.Coordinator` object.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
    """
    _WrappedSession.__init__(self, sess)
    self._coord = coord
    self._stop_grace_period_secs = stop_grace_period_secs

  def _check_stop(self):
    # Check with the coordinator if we should stop.
    return self._coord.should_stop()

  def close(self):
    self._coord.request_stop()
    try:
      self._coord.join(
          stop_grace_period_secs=self._stop_grace_period_secs,
          ignore_live_threads=True)
    finally:
      try:
        _WrappedSession.close(self)
      except Exception:  # pylint: disable=broad-except
        # We intentionally suppress exceptions from the close() here since
        # useful exceptions are already reported by join().
        pass


class _HookedSession(_WrappedSession):
  """A _WrappedSession that calls hooks during calls to run().

  The list of hooks to call is passed in the constructor.  Before each call
  to `run()` the session calls the `before_run()` method of the hooks, which
  can return additional ops or tensors to run.  These are added to the arguments
  of the call to `run()`.

  When the `run()` call finishes, the session calls the `after_run()` methods of
  the hooks, passing the values returned by the `run()` call corresponding to
  the ops and tensors that each hook requested.

  If any call to the hooks, requests stop via run_context the session will be
  marked as needing to stop and its `should_stop()` method will now return
  `True`.
  """

  def __init__(self, sess, hooks):
    """Initializes a _HookedSession object.

    Args:
      sess: A `tf.Session` or a `_WrappedSession` object.
      hooks: An iterable of `SessionRunHook' objects.
    """

    _WrappedSession.__init__(self, sess)
    self._hooks = hooks
    self._should_stop = False

  def _check_stop(self):
    """See base class."""
    return self._should_stop

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """See base class."""
    if self.should_stop():
      raise RuntimeError('Run called even after should_stop requested.')

    actual_fetches = {'caller': fetches}

    run_context = session_run_hook.SessionRunContext(
        original_args=session_run_hook.SessionRunArgs(fetches, feed_dict),
        session=self._sess)

    options = options or config_pb2.RunOptions()
    feed_dict = self._call_hook_before_run(run_context, actual_fetches,
                                           feed_dict, options)

    # Do session run.
    run_metadata = run_metadata or config_pb2.RunMetadata()
    outputs = _WrappedSession.run(self,
                                  fetches=actual_fetches,
                                  feed_dict=feed_dict,
                                  options=options,
                                  run_metadata=run_metadata)

    for hook in self._hooks:
      hook.after_run(
          run_context,
          session_run_hook.SessionRunValues(
              results=outputs[hook] if hook in outputs else None,
              options=options,
              run_metadata=run_metadata))
    self._should_stop = self._should_stop or run_context.stop_requested

    return outputs['caller']

  def _call_hook_before_run(self, run_context, fetch_dict, user_feed_dict,
                            options):
    """Calls hooks.before_run and handles requests from hooks."""
    hook_feeds = {}
    for hook in self._hooks:
      request = hook.before_run(run_context)
      if request is not None:
        if request.fetches is not None:
          fetch_dict[hook] = request.fetches
        if request.feed_dict:
          self._raise_if_feeds_intersects(
              hook_feeds, request.feed_dict,
              'Same tensor is fed by two hooks.')
          hook_feeds.update(request.feed_dict)
        if request.options:
          self._merge_run_options(options, request.options)

    if not hook_feeds:
      return user_feed_dict

    if not user_feed_dict:
      return hook_feeds

    self._raise_if_feeds_intersects(
        user_feed_dict, hook_feeds,
        'Same tensor is fed by a SessionRunHook and user.')
    hook_feeds.update(user_feed_dict)
    return hook_feeds

  def _raise_if_feeds_intersects(self, feeds1, feeds2, message):
    intersection = set(feeds1.keys()) & set(feeds2.keys())
    if intersection:
      raise RuntimeError(message + ' Conflict(s): ' + str(list(intersection)))

  def _merge_run_options(self, options, incoming_options):
    """Merge two instances of RunOptions into the first one.

    During the merger, the numerical fields including trace_level,
    timeout_in_ms, inter_op_thread_pool are set to the larger one of the two.
    The boolean value is set to the logical OR of the two.
    debug_tensor_watch_opts of the original options is extended with that from
    the incoming one.

    Args:
      options: The options to merge into.
      incoming_options: The options to be merged into the first argument.
    """
    options.trace_level = max(options.trace_level, incoming_options.trace_level)
    options.timeout_in_ms = max(options.timeout_in_ms,
                                incoming_options.timeout_in_ms)
    options.inter_op_thread_pool = max(options.inter_op_thread_pool,
                                       incoming_options.inter_op_thread_pool)
    options.output_partition_graphs = max(
        options.output_partition_graphs,
        incoming_options.output_partition_graphs)

    options.debug_options.debug_tensor_watch_opts.extend(
        incoming_options.debug_options.debug_tensor_watch_opts)
