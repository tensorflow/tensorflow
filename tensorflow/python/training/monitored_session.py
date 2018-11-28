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
import sys

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
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
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export


# The list of exceptions that we should recover from. Exceptions not in this
# list may terminate the job.
_PREEMPTION_ERRORS = (errors.AbortedError, errors.UnavailableError)


# Value that indicates no value was provided.
USE_DEFAULT = object()


@tf_export('train.Scaffold')
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

  * `saver`: A `tf.train.Saver` object taking care of saving the variables.
    Picked from and stored into the `SAVERS` collection in the graph by default.
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

  * `init_feed_dict`: A session feed dictionary that should be used when
     running the init op.
  * `init_fn`: A callable to run after the init op to perform additional
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
               saver=None,
               copy_from_scaffold=None):
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
      saver: Optional `tf.train.Saver` object to use to save and restore
        variables.
      copy_from_scaffold: Optional scaffold object to copy fields from. Its
        fields will be overwritten by the provided fields in this function.
    """
    if copy_from_scaffold is not None:
      if not isinstance(copy_from_scaffold, Scaffold):
        raise TypeError('copy_from_scaffold is not a Scaffold instance.')
      # We need _coalesce since Tensor is not converted to bool automatically,
      # so the common idiom of (a or b) does not work.
      coalesce = lambda a, b: a if a is not None else b
      init_op = coalesce(init_op, copy_from_scaffold.init_op)
      init_feed_dict = coalesce(init_feed_dict,
                                copy_from_scaffold.init_feed_dict)
      # Use the original init_fn provided by the user to init the new Scaffold.
      init_fn = coalesce(init_fn, copy_from_scaffold._user_init_fn)  # pylint: disable=protected-access
      ready_op = coalesce(ready_op, copy_from_scaffold.ready_op)
      ready_for_local_init_op = coalesce(
          ready_for_local_init_op, copy_from_scaffold.ready_for_local_init_op)
      local_init_op = coalesce(local_init_op, copy_from_scaffold.local_init_op)
      summary_op = coalesce(summary_op, copy_from_scaffold.summary_op)
      saver = coalesce(saver, copy_from_scaffold.saver)

    # NOTE(touts): modifying the init function to be passed the scaffold is a
    # hack to make it easy to find the saver.  Is there a better way?
    self._user_init_fn = init_fn
    if init_fn:
      self._init_fn = lambda sess: init_fn(self, sess)
    else:
      self._init_fn = None

    self._init_op = init_op
    self._init_feed_dict = init_feed_dict
    self._ready_op = ready_op
    self._ready_for_local_init_op = ready_for_local_init_op
    self._local_init_op = local_init_op
    self._summary_op = summary_op
    self._saver = saver

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
        return array_ops.concat([
            variables.report_uninitialized_variables(
                variables.global_variables()),
            resources.report_uninitialized_resources(
                resources.shared_resources())
        ], 0)
      self._ready_for_local_init_op = Scaffold.get_or_default(
          'ready_for_local_init_op', ops.GraphKeys.READY_FOR_LOCAL_INIT_OP,
          default_ready_for_local_init_op)
    if self._local_init_op is None:
      self._local_init_op = Scaffold.get_or_default(
          'local_init_op', ops.GraphKeys.LOCAL_INIT_OP,
          Scaffold.default_local_init_op)
    if self._summary_op is None:
      self._summary_op = Scaffold.get_or_default('summary_op',
                                                 ops.GraphKeys.SUMMARY_OP,
                                                 summary.merge_all)
    # pylint: disable=g-long-lambda
    if self._saver is None:
      self._saver = training_saver._get_saver_or_default()  # pylint: disable=protected-access
    # pylint: enable=g-long-lambda
    self._saver.build()

    ops.get_default_graph().finalize()
    logging.info('Graph was finalized.')
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
  def default_local_init_op():
    """Returns an op that groups the default local init ops.

    This op is used during session initialization when a Scaffold is
    initialized without specifying the local_init_op arg. It includes
    `tf.local_variables_initializer`, `tf.tables_initializer`, and also
    initializes local session resources.

    Returns:
      The default Scaffold local init op.
    """
    return control_flow_ops.group(
        variables.local_variables_initializer(),
        lookup_ops.tables_initializer(),
        resources.initialize_resources(resources.local_resources()))


def _create_monitored_session_with_worker_context(worker_context,  # pylint: disable=missing-docstring
                                                  scaffold,
                                                  checkpoint_dir=None,
                                                  hooks=None,
                                                  chief_only_hooks=None,
                                                  save_checkpoint_secs=None,
                                                  save_summaries_steps=None,
                                                  save_summaries_secs=None,
                                                  config=None,
                                                  stop_grace_period_secs=120,
                                                  log_step_count_steps=100,
                                                  max_wait_secs=7200,
                                                  save_checkpoint_steps=None,
                                                  summary_dir=None):
  all_hooks = []
  if hooks:
    all_hooks.extend(hooks)
  if chief_only_hooks and worker_context.is_chief:
    all_hooks.extend(chief_only_hooks)

  summary_dir = summary_dir or checkpoint_dir
  if summary_dir and worker_context.should_save_summary:
    if log_step_count_steps and log_step_count_steps > 0:
      all_hooks.append(
          basic_session_run_hooks.StepCounterHook(
              output_dir=summary_dir, every_n_steps=log_step_count_steps))

    if (save_summaries_steps and save_summaries_steps > 0) or (
        save_summaries_secs and save_summaries_secs > 0):
      all_hooks.append(
          basic_session_run_hooks.SummarySaverHook(
              scaffold=scaffold,
              save_steps=save_summaries_steps,
              save_secs=save_summaries_secs,
              output_dir=summary_dir))

  if checkpoint_dir and worker_context.should_checkpoint:
    if (save_checkpoint_secs and save_checkpoint_secs > 0) or (
        save_checkpoint_steps and save_checkpoint_steps > 0):
      all_hooks.append(
          basic_session_run_hooks.CheckpointSaverHook(
              checkpoint_dir,
              save_steps=save_checkpoint_steps,
              save_secs=save_checkpoint_secs,
              scaffold=scaffold))

  session_creator = worker_context.session_creator(
      scaffold,
      config=config,
      checkpoint_dir=checkpoint_dir,
      max_wait_secs=max_wait_secs)
  return MonitoredSession(
      session_creator=session_creator,
      hooks=all_hooks,
      stop_grace_period_secs=stop_grace_period_secs)


@tf_export(v1=['train.MonitoredTrainingSession'])
def MonitoredTrainingSession(master='',  # pylint: disable=invalid-name
                             is_chief=True,
                             checkpoint_dir=None,
                             scaffold=None,
                             hooks=None,
                             chief_only_hooks=None,
                             save_checkpoint_secs=USE_DEFAULT,
                             save_summaries_steps=USE_DEFAULT,
                             save_summaries_secs=USE_DEFAULT,
                             config=None,
                             stop_grace_period_secs=120,
                             log_step_count_steps=100,
                             max_wait_secs=7200,
                             save_checkpoint_steps=USE_DEFAULT,
                             summary_dir=None):
  """Creates a `MonitoredSession` for training.

  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  initialize/restore. Please check `tf.train.MonitoredSession` for more
  information.


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
      using a default checkpoint saver. If both `save_checkpoint_steps` and
      `save_checkpoint_secs` are set to `None`, then the default checkpoint
      saver isn't used. If both are provided, then only `save_checkpoint_secs`
      is used. Default 600.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.
    max_wait_secs: Maximum time workers should wait for the session to
      become available. This should be kept relatively short to help detect
      incorrect code, but sometimes may need to be increased if the chief takes
      a while to start up.
    save_checkpoint_steps: The frequency, in number of global steps, that a
      checkpoint is saved using a default checkpoint saver. If both
      `save_checkpoint_steps` and `save_checkpoint_secs` are set to `None`, then
      the default checkpoint saver isn't used. If both are provided, then only
      `save_checkpoint_secs` is used. Default not enabled.
    summary_dir: A string.  Optional path to a directory where to
      save summaries. If None, checkpoint_dir is used instead.

  Returns:
    A `MonitoredSession` object.
  """
  if save_summaries_steps == USE_DEFAULT and save_summaries_secs == USE_DEFAULT:
    save_summaries_steps = 100
    save_summaries_secs = None
  elif save_summaries_secs == USE_DEFAULT:
    save_summaries_secs = None
  elif save_summaries_steps == USE_DEFAULT:
    save_summaries_steps = None

  if (save_checkpoint_steps == USE_DEFAULT and
      save_checkpoint_secs == USE_DEFAULT):
    save_checkpoint_steps = None
    save_checkpoint_secs = 600
  elif save_checkpoint_secs == USE_DEFAULT:
    save_checkpoint_secs = None
  elif save_checkpoint_steps == USE_DEFAULT:
    save_checkpoint_steps = None

  scaffold = scaffold or Scaffold()
  worker_context = distribute_coordinator_context.get_current_worker_context()

  if worker_context:
    return _create_monitored_session_with_worker_context(
        worker_context,
        scaffold,
        checkpoint_dir=checkpoint_dir,
        hooks=hooks,
        chief_only_hooks=chief_only_hooks,
        save_checkpoint_secs=save_checkpoint_secs,
        save_summaries_steps=save_summaries_steps,
        save_summaries_secs=save_summaries_secs,
        config=config,
        stop_grace_period_secs=stop_grace_period_secs,
        log_step_count_steps=log_step_count_steps,
        max_wait_secs=max_wait_secs,
        save_checkpoint_steps=save_checkpoint_steps,
        summary_dir=summary_dir)

  if not is_chief:
    session_creator = WorkerSessionCreator(
        scaffold=scaffold,
        master=master,
        config=config,
        max_wait_secs=max_wait_secs)
    return MonitoredSession(
        session_creator=session_creator,
        hooks=hooks or [],
        stop_grace_period_secs=stop_grace_period_secs)

  all_hooks = []
  if chief_only_hooks:
    all_hooks.extend(chief_only_hooks)
  session_creator = ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_dir=checkpoint_dir,
      master=master,
      config=config)

  summary_dir = summary_dir or checkpoint_dir
  if summary_dir:
    if log_step_count_steps and log_step_count_steps > 0:
      all_hooks.append(
          basic_session_run_hooks.StepCounterHook(
              output_dir=summary_dir, every_n_steps=log_step_count_steps))

    if (save_summaries_steps and save_summaries_steps > 0) or (
        save_summaries_secs and save_summaries_secs > 0):
      all_hooks.append(
          basic_session_run_hooks.SummarySaverHook(
              scaffold=scaffold,
              save_steps=save_summaries_steps,
              save_secs=save_summaries_secs,
              output_dir=summary_dir))

  if checkpoint_dir:
    if (save_checkpoint_secs and save_checkpoint_secs > 0) or (
        save_checkpoint_steps and save_checkpoint_steps > 0):
      all_hooks.append(
          basic_session_run_hooks.CheckpointSaverHook(
              checkpoint_dir,
              save_steps=save_checkpoint_steps,
              save_secs=save_checkpoint_secs,
              scaffold=scaffold))

  if hooks:
    all_hooks.extend(hooks)
  return MonitoredSession(
      session_creator=session_creator,
      hooks=all_hooks,
      stop_grace_period_secs=stop_grace_period_secs)


@tf_export(v1=['train.SessionCreator'])
@six.add_metaclass(abc.ABCMeta)
class SessionCreator(object):
  """A factory for tf.Session."""

  @abc.abstractmethod
  def create_session(self):
    raise NotImplementedError(
        'create_session is not implemented for {}.'.format(self))


@tf_export(v1=['train.ChiefSessionCreator'])
class ChiefSessionCreator(SessionCreator):
  """Creates a tf.Session for a chief."""

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


@tf_export(v1=['train.WorkerSessionCreator'])
class WorkerSessionCreator(SessionCreator):
  """Creates a tf.Session for a worker."""

  def __init__(self,
               scaffold=None,
               master='',
               config=None,
               max_wait_secs=30 * 60):
    """Initializes a worker session creator.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
      max_wait_secs: Maximum time to wait for the session to become available.
    """
    self._scaffold = scaffold or Scaffold()
    self._session_manager = None
    self._master = master
    self._config = config
    self._max_wait_secs = max_wait_secs

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
        self._master, config=self._config,
        max_wait_secs=self._max_wait_secs
    )


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
        and `UnavailableError` or not.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
    """
    self._graph_was_finalized = ops.get_default_graph().finalized
    self._hooks = hooks or []
    for h in self._hooks:
      h.begin()

    worker_context = distribute_coordinator_context.get_current_worker_context()
    if not session_creator and worker_context:
      session_creator = worker_context.session_creator()

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

  def run_step_fn(self, step_fn):
    """Run ops using a step function.

    Args:
      step_fn: A function or a method with a single argument of type
        `StepContext`.  The function may use methods of the argument to
        perform computations with access to a raw session.

        The returned value of the `step_fn` will be returned from `run_step_fn`,
        unless a stop is requested.  In that case, the next `should_stop` call
        will return True.

        Example usage:

        ```python
           with tf.Graph().as_default():
             c = tf.placeholder(dtypes.float32)
             v = tf.add(c, 4.0)
             w = tf.add(c, 0.5)

             def step_fn(step_context):
               a = step_context.session.run(fetches=v, feed_dict={c: 0.5})
               if a <= 4.5:
                 step_context.request_stop()
               return step_context.run_with_hooks(fetches=w, feed_dict={c: 0.1})

             with tf.MonitoredSession() as session:
               while not session.should_stop():
                 a = session.run_step_fn(step_fn)
        ```

        Hooks interact with the `run_with_hooks()` call inside the `step_fn`
        as they do with a `MonitoredSession.run` call.

    Returns:
      Returns the returned value of `step_fn`.

    Raises:
      StopIteration: if `step_fn` has called `request_stop()`.  It may be
        caught by `with tf.MonitoredSession()` to close the session.
      ValueError: if `step_fn` doesn't have a single argument called
        `step_context`. It may also optionally have `self` for cases when it
        belongs to an object.
    """
    step_fn_arguments = function_utils.fn_args(step_fn)
    if step_fn_arguments != ('step_context',) and step_fn_arguments != (
        'self',
        'step_context',
    ):
      raise ValueError(
          '`step_fn` may either have one `step_context` argument, or'
          ' `self` and `step_context` arguments if it\'s an instance'
          ' method. Got {} instead.'.format(step_fn_arguments))

    # `self._sess` is either `_RecoverableSession` or a `_CoordinatedSession`.
    # Setting `run_with_hooks` to `None` will cause `run_with_hooks` to be
    # `_CoordinatedSession.run` downstream in either case. This allows
    # `_PREEMPTION_ERRORS` to propage from within `step_fn` to
    # `_RecoverableSession.run_step_fn`.
    return self._sess.run_step_fn(step_fn, self._tf_sess(), run_with_hooks=None)

  class StepContext(object):
    """Control flow instrument for the `step_fn` from `run_step_fn()`.

       Users of `step_fn` may perform `run()` calls without running hooks
       by accessing the `session`.  A `run()` call with hooks may be performed
       using `run_with_hooks()`.  Computation flow can be interrupted using
       `request_stop()`.
    """

    def __init__(self, session, run_with_hooks_fn):
      """Initializes the `step_context` argument for a `step_fn` invocation.

      Args:
        session: An instance of `tf.Session`.
        run_with_hooks_fn: A function for running fetches and hooks.
      """
      self._session = session
      self._run_with_hooks_fn = run_with_hooks_fn

    @property
    def session(self):
      return self._session

    def run_with_hooks(self, *args, **kwargs):
      """Same as `MonitoredSession.run`. Accepts the same arguments."""
      return self._run_with_hooks_fn(*args, **kwargs)

    def request_stop(self):
      """Exit the training loop by causing `should_stop()` to return `True`.

         Causes `step_fn` to exit by raising an exception.

      Raises:
        StopIteration
      """
      raise StopIteration('step_fn has requested the iterations to stop.')

  def should_stop(self):
    return self._sess is None or self._sess.should_stop()

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

  class _CoordinatedSessionCreator(SessionCreator):
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
      if ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
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
        if self._sess is None:
          raise RuntimeError('Session is already closed.')
        self._sess.close()
      finally:
        self._sess = None
        self._coordinated_creator.tf_sess = None
        self._coordinated_creator.coord = None
        if not self._graph_was_finalized:
          ops.get_default_graph()._unsafe_unfinalize()  # pylint: disable=protected-access

  def _is_closed(self):
    """Return True if the monitored session is closed.  For tests only.

    Returns:
      A boolean.
    """
    return self._coordinated_creator.tf_sess is None

  def _tf_sess(self):
    """Return underlying tf.Session object.

    Warning: accessing the returned object in user code is likely to cause races
    or "flaky tests".

    Returns:
      A tf.Session object.
    """
    return self._coordinated_creator.tf_sess


@tf_export(v1=['train.MonitoredSession'])
class MonitoredSession(_MonitoredSession):
  """Session-like object that handles initialization, recovery and hooks.

  Example usage:

  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummarySaverHook(...)
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
  * calls `hook.after_create_session()`

  Run: When `run()` is called, the monitored session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user
  * if `AbortedError` or `UnavailableError` occurs, it recovers or
    reinitializes the session before executing the run() call again


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

  Note: This is not a `tf.Session`. For example, it cannot do following:

  * it cannot be set as default session.
  * it cannot be sent to saver.save.
  * it cannot be sent to tf.train.start_queue_runners.

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


@tf_export(v1=['train.SingularMonitoredSession'])
class SingularMonitoredSession(_MonitoredSession):
  """Session-like object that handles initialization, restoring, and hooks.

  Please note that this utility is not recommended for distributed settings.
  For distributed settings, please use `tf.train.MonitoredSession`. The
  differences between `MonitoredSession` and `SingularMonitoredSession` are:

  * `MonitoredSession` handles `AbortedError` and `UnavailableError` for
    distributed settings, but `SingularMonitoredSession` does not.
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
  summary_hook = SummarySaverHook(...)
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
  * suppresses `OutOfRange` error which indicates that all inputs have been
    processed if the `SingularMonitoredSession` is used as a context.
  """

  def __init__(self,
               hooks=None,
               scaffold=None,
               master='',
               config=None,
               checkpoint_dir=None,
               stop_grace_period_secs=120,
               checkpoint_filename_with_path=None):
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
      checkpoint_filename_with_path: A string. Optional path to a checkpoint
        file from which to restore variables.
    """
    session_creator = ChiefSessionCreator(
        scaffold=scaffold,
        master=master,
        config=config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename_with_path=checkpoint_filename_with_path)
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
      except _PREEMPTION_ERRORS as e:
        logging.warning('An error occurred when attempting to close the '
                        'session. This may be due to a preemption in a '
                        'connected worker or parameter server. Error: %s', e)
      finally:
        self._sess = None

  def run(self, *args, **kwargs):
    return self._sess.run(*args, **kwargs)

  def run_step_fn(self, step_fn, raw_session, run_with_hooks):
    # `_RecoverableSession` sets `run_with_hooks` to `_CoordinatedSession.run`.
    # It is `None` when called from `_CoordinatedSession`. In that case
    # `self.run` is `_CoordinatedSession.run`.
    run_with_hooks = run_with_hooks or self.run
    return step_fn(_MonitoredSession.StepContext(raw_session, run_with_hooks))


class _RecoverableSession(_WrappedSession):
  """A wrapped session that recreates a session upon certain kinds of errors.

  The constructor is passed a SessionCreator object, not a session.

  Calls to `run()` are delegated to the wrapped session.  If a call raises the
  exception `tf.errors.AbortedError` or `tf.errors.UnavailableError`, the
  wrapped session is closed, and a new one is created by calling the factory
  again.
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
      except _PREEMPTION_ERRORS as e:
        logging.info('An error was raised while a session was being created. '
                     'This may be due to a preemption of a connected worker '
                     'or parameter server. A new session will be created. '
                     'This error may also occur due to a gRPC failure caused '
                     'by high memory or network bandwidth usage in the '
                     'parameter servers. If this error occurs repeatedly, try '
                     'increasing the number of parameter servers assigned to '
                     'the job. Error: %s', e)

  def _check_stop(self):
    try:
      if self._sess:
        return self._sess._check_stop()  # pylint: disable=protected-access
      else:
        return True
    except _PREEMPTION_ERRORS as e:
      logging.info('An error was raised while considering whether the '
                   'session is complete. This may be due to a preemption in '
                   'a connected worker or parameter server. The current '
                   'session will be closed and a new session will be '
                   'created. This error may also occur due to a gRPC failure '
                   'caused by high memory or network bandwidth usage in the '
                   'parameter servers. If this error occurs repeatedly, try '
                   'increasing the number of parameter servers assigned to '
                   'the job. Error: %s', e)
      self.close()
      self._sess = self._create_session()
      # Since we have just recreated the session, the overall computation should
      # not stop:
      return False
    except Exception:  # pylint: disable=broad-except
      # `should_stop` should return True instead of raising an exception.
      return True

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    while True:
      try:
        if not self._sess:
          self._sess = self._create_session()
        return self._sess.run(fetches,
                              feed_dict=feed_dict,
                              options=options,
                              run_metadata=run_metadata)
      except _PREEMPTION_ERRORS as e:
        logging.info('An error was raised. This may be due to a preemption in '
                     'a connected worker or parameter server. The current '
                     'session will be closed and a new session will be '
                     'created. This error may also occur due to a gRPC failure '
                     'caused by high memory or network bandwidth usage in the '
                     'parameter servers. If this error occurs repeatedly, try '
                     'increasing the number of parameter servers assigned to '
                     'the job. Error: %s', e)
        self.close()
        self._sess = None

  def run_step_fn(self, step_fn, raw_session, run_with_hooks):
    while True:
      try:
        if not self._sess:
          self._sess = self._create_session()

        run_with_hooks = self._sess.run
        return self._sess.run_step_fn(step_fn, raw_session, run_with_hooks)
      except _PREEMPTION_ERRORS as e:
        logging.info('An error was raised. This may be due to a preemption in '
                     'a connected worker or parameter server. The current '
                     'session will be closed and a new session will be '
                     'created. This error may also occur due to a gRPC failure '
                     'caused by high memory or network bandwidth usage in the '
                     'parameter servers. If this error occurs repeatedly, try '
                     'increasing the number of parameter servers assigned to '
                     'the job. Error: %s', e)
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
    # If the coordinator was asked to stop due to an exception, then it needs
    # to be propagated to this stack.
    self._coord.raise_requested_exception()
    # At this point, no exceptions are recorded in the coordinator.
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

  def run(self, *args, **kwargs):
    try:
      return self._sess.run(*args, **kwargs)
    except _PREEMPTION_ERRORS:
      raise
    except Exception:  # pylint: disable=broad-except
      # A non-preemption error could have been caused by a preemption error
      # in the coordinator. If this is the case, raise that exception instead,
      # since it's the root cause. Otherwise, stick to the `original_exc_info`.
      original_exc_info = sys.exc_info()
      try:
        self._coord.raise_requested_exception()
      except _PREEMPTION_ERRORS:
        raise
      except Exception:  # pylint: disable=broad-except
        raise six.reraise(*original_exc_info)
      else:
        raise six.reraise(*original_exc_info)


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
    options.debug_options.reset_disk_byte_usage = (
        options.debug_options.reset_disk_byte_usage or
        incoming_options.debug_options.reset_disk_byte_usage)
