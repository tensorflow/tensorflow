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
"""Wrapper for a Session-like object that handles threads and recovery.

Based on an original design of Illia Polosukhin.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import coordinated_session
from tensorflow.contrib.learn.python.learn import monitored_session
from tensorflow.contrib.learn.python.learn import recoverable_session
from tensorflow.contrib.learn.python.learn import summary_writer_cache
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import training_util


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

  If you call the scaffold constructor without any arguments it will pick
  pieces from the collections, creating default ones if needed.  You can pass
  arguments to the constructor to provide your own pieces.  Pieces that you
  pass to the constructor are not added to the graph collections.

  The following pieces are directly accessible as attributes of the `Scaffold`
  object:

  * `saver`: A `tf.Saver` object taking care of saving the variables.  Picked
    from and stored into the `SAVERS` collection in the graph.
  * `init_op`: An op to run to initialize the variables.  Picked from and
    stored into the `INIT_OP` collection in the graph.
  * `ready_op`: An op to verify that the variables are initialized.  Picked
    from and stored into the `READY_OP` collection in the graph.
  * `local_init_op`: An op to initialize the local variables.  Picked
    from and stored into the `LOCAL_INIT_OP` collection in the graph.
  * `summary_op`: An op to run and merge the summaries in the graph.  Picked
    from and stored into the `SUMMARY_OP` collection in the graph.
  * `global_step`: A tensor containing the global step counter.  Picked
    from and stored into the `GLOBAL_STEP` collection in the graph.

  You can also pass the following additional pieces to the constructor:

  * `init_feed_dict`: A sessionn feed dictionary that should be used when
     running the init op.
  * `init_fn`: A callable to run run after the init op to perform additional
    initializations.  The callable will be called as
    `init_fn(scaffold, session)`.

  """
  # TODO(touts): consider adding the output dir and summary writer (cached)?
  # TODO(touts): I do not think we should pass keep_checkpoint_max here.
  # TODO(touts): Add individual static functions for init_op(), etc. that
  # implement the caching logic.

  def __init__(self,
               global_step_tensor=None,
               init_op=None,
               init_feed_dict=None,
               init_fn=None,
               ready_op=None,
               local_init_op=None,
               summary_op=None,
               saver=None,
               keep_checkpoint_max=5):
    """Create a scaffold.

    Args:
      global_step_tensor: Optional tensor to use as the global step counter.
      init_op: Optional op for initializing variables.
      init_feed_dict: Optional session feed dictionary to use when running the
        init_op.
      init_fn: Optional function to use to initialize the model after running
        the init_op.  Will be called as `init_fn(scaffold, session)`.
      ready_op: Optional op to verify that the variables are initialized.  Must
        return an empty scalar string tensor when the variables are
        initialized, or a non-empty one listing the names of the
        non-initialized variables.
      local_init_op: Optional op to initialize local variables.
      summary_op: Optional op to gather all summaries.  Must return a scalar
        string tensor containing a serialized `Summary` proto.
      saver: Optional `tf.Saver` object to use to save and restore variables.
      keep_checkpoint_max: Optional parameter to use to construct a saver if
        none is already there in the graph.
    """

    # NOTE(touts): modifying the init function to be passed the scaffold is a
    # hack to make it easy to find the saver.  Is there a better way?
    if init_fn:
      self._init_fn = lambda sess: init_fn(self, sess)
    else:
      self._init_fn = None

    self._global_step_tensor = global_step_tensor
    self._init_op = init_op
    self._ready_op = ready_op
    self._local_init_op = local_init_op
    self._summary_op = summary_op
    self._saver = saver
    self._keep_checkpoint_max = keep_checkpoint_max
    self._init_feed_dict = init_feed_dict

  def finalize(self):
    """Creates operations if needed and finalizes the graph."""
    if self._global_step_tensor is None:
      self._global_step_tensor = contrib_variables.get_or_create_global_step()
    if self._init_op is None:
      self._init_op = Scaffold._get_or_default(
          'init_op', ops.GraphKeys.INIT_OP, variables.initialize_all_variables)
    if self._ready_op is None:
      self._ready_op = Scaffold._get_or_default(
          'ready_op', ops.GraphKeys.READY_OP,
          variables.report_uninitialized_variables)
    if self._local_init_op is None:
      self._local_init_op = Scaffold._get_or_default(
          'local_init_op', ops.GraphKeys.LOCAL_INIT_OP,
          Scaffold._default_local_init_op)
    if self._summary_op is None:
      self._summary_op = Scaffold._get_or_default(
          'summary_op', ops.GraphKeys.SUMMARY_OP,
          logging_ops.merge_all_summaries)
    # pylint: disable=g-long-lambda
    if self._saver is None:
      self._saver = Scaffold._get_or_default(
          'saver',
          ops.GraphKeys.SAVERS,
          lambda: training_saver.Saver(sharded=True,
                                       max_to_keep=self._keep_checkpoint_max))
    # pylint: enable=g-long-lambda

    ops.get_default_graph().finalize()

  @property
  def global_step_tensor(self):
    return self._global_step_tensor

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
  def _get_or_default(arg_name, collection_key, default_constructor):
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
    return control_flow_ops.group(variables.initialize_local_variables(),
                                  data_flow_ops.initialize_all_tables())


def _call_monitor_end(monitor, sess):
  # TODO(ispir): Remove following check when switch to MonitorV2
  if 'session' in inspect.getargspec(monitor.end).args:
    monitor.end(session=sess)
  else:
    monitor.end()


# TODO(ispir): Document this class after interface is finalized.
# mention StopIteration and OutOfRangeError
class SupervisedSession(object):
  """Session-like object that supports recovery and monitors.


  """

  def __init__(self,
               master,
               is_chief=True,
               checkpoint_dir=None,
               monitors=None,
               scaffold=None,
               config=None):
    self._graph = ops.get_default_graph()
    self._master = master
    self._checkpoint_dir = checkpoint_dir
    self._is_chief = is_chief
    self._config = config
    self._monitors = monitors or []
    self._scaffold = scaffold or Scaffold()
    for monitor in self._monitors:
      monitor.begin(max_steps=None)
    # Create the session.
    self._scaffold.finalize()
    self._session_manager = sm.SessionManager(
        local_init_op=self._scaffold.local_init_op,
        ready_op=self._scaffold.ready_op,
        graph=ops.get_default_graph())
    self._sess = recoverable_session.RecoverableSession(self._create_session)
    # Call the begin() method of monitors.
    self._init_step = self._tf_sess.run(self._scaffold.global_step_tensor)
    # Write the graph out, note: this uses self._init_step.
    self.write_graph()

  def _create_session(self):
    """Factory for the RecoverableSession.

    Returns:
      A session, initialized or recovered as needed.
    """
    if self._is_chief:
      tf_sess = self._session_manager.prepare_session(
          self._master, saver=self._scaffold.saver,
          checkpoint_dir=self._checkpoint_dir, config=self._config,
          init_op=self._scaffold.init_op,
          init_feed_dict=self._scaffold.init_feed_dict,
          init_fn=self._scaffold.init_fn)
    else:
      tf_sess = self._session_manager.wait_for_session(
          self._master, config=self._config)
    # Keep the tf_sess for quick runs of global step when needed.
    self._tf_sess = tf_sess
    # We don't want coordinator to suppress any exception.
    coord = coordinator.Coordinator(clean_stop_exception_types=[])
    coordinated_threads_to_join = queue_runner.start_queue_runners(sess=tf_sess,
                                                                   coord=coord)
    return coordinated_session.CoordinatedSession(
        monitored_session.MonitoredSession(tf_sess, self._monitors,
                                           self._scaffold.global_step_tensor),
        coord, coordinated_threads_to_join)

  @property
  def scaffold(self):
    return self._scaffold

  @property
  def session(self):
    return self._tf_sess

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Run ops in the supervised session.

    This method is completely compatible with the `tf.Session.run()` method.

    Args:
      fetches: Same as `tf.Session.run()`.
      feed_dict: Same as `tf.Session.run()`.
      options: Same as `tf.Session.run()`.
      run_metadata: Same as `tf.Session.run()`.

    Returns:
      Same as `tf.Session.run()`.
    """
    return self._sess.run(fetches, feed_dict=feed_dict, options=options,
                          run_metadata=run_metadata)

  def should_stop(self):
    if self._sess:
      return self._sess.should_stop()
    return True

  def close(self):
    self._close_internal()

  def _close_internal(self, exception_type=None):
    try:
      if not exception_type:
        for monitor in self._monitors:
          _call_monitor_end(monitor, self._tf_sess)
    finally:
      self._sess.close()
      self._sess = None
      self._tf_sess = None

  def _is_closed(self):
    """Return True if the supervised session is closed.  For tests only.

    Returns:
      A boolean.
    """
    return self._tf_sess is None

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    if exception_type in [errors.OutOfRangeError, StopIteration]:
      # TODO(ispir): log error if Coordinator hasn't done already.
      exception_type = None
    self._close_internal(exception_type)
    # __exit__ should return True to suppress an exception.
    return exception_type is None

  def write_graph(self):
    """Saves current graph."""
    if self._checkpoint_dir is not None and self._is_chief:
      summary_writer = summary_writer_cache.SummaryWriterCache.get(
          self._checkpoint_dir)
      training_util.write_graph(self._graph.as_graph_def(add_shapes=True),
                                self._checkpoint_dir, 'graph.pbtxt')
      summary_writer.add_graph(self._graph)
      summary_writer.add_session_log(SessionLog(status=SessionLog.START),
                                     self._init_step)
