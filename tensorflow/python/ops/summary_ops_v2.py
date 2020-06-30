# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Operations to emit summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import getpass
import os
import re
import threading
import time

import six

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

# Name for graph collection of summary writer init ops, which is only exposed
# as a legacy API for tf.contrib.summary in TF 1.x.
_SUMMARY_WRITER_INIT_COLLECTION_NAME = "_SUMMARY_WRITER_V2"

_EXPERIMENT_NAME_PATTERNS = re.compile(r"^[^\x00-\x1F<>]{0,256}$")
_RUN_NAME_PATTERNS = re.compile(r"^[^\x00-\x1F<>]{0,512}$")
_USER_NAME_PATTERNS = re.compile(r"^[a-z]([-a-z0-9]{0,29}[a-z0-9])?$", re.I)


class _SummaryState(threading.local):

  def __init__(self):
    super(_SummaryState, self).__init__()
    self.is_recording = None
    # TODO(slebedev): why a separate flag for DS and is it on by default?
    self.is_recording_distribution_strategy = True
    self.writer = None
    self.step = None


_summary_state = _SummaryState()


def _should_record_summaries_internal(default_state):
  """Returns boolean Tensor if summaries should/shouldn't be recorded.

  Now the summary condition is decided by logical "and" of below conditions:
  First, summary writer must be set. Given this constraint is met,
  ctx.summary_recording and ctx.summary_recording_distribution_strategy.
  The former one is usually set by user, and the latter one is controlled
  by DistributionStrategy (tf.distribute.ReplicaContext).

  Args:
    default_state: can be True or False. The default summary behavior when
    summary writer is set and the user does not specify
    ctx.summary_recording and ctx.summary_recording_distribution_strategy
    is True.
  """
  if _summary_state.writer is None:
    return constant_op.constant(False)

  if not callable(_summary_state.is_recording):
    static_cond = tensor_util.constant_value(_summary_state.is_recording)
    if static_cond is not None and not static_cond:
      return constant_op.constant(False)

  resolve = lambda x: x() if callable(x) else x
  cond_distributed = resolve(_summary_state.is_recording_distribution_strategy)
  cond = resolve(_summary_state.is_recording)
  if cond is None:
    cond = default_state
  return math_ops.logical_and(cond_distributed, cond)


def _should_record_summaries_v2():
  """Returns boolean Tensor which is true if summaries should be recorded.

  If no recording status has been set, this defaults to True, unlike the public
  should_record_summaries().
  """
  return _should_record_summaries_internal(default_state=True)


@tf_export("summary.should_record_summaries", v1=[])
def should_record_summaries():
  """Returns boolean Tensor which is true if summaries should be recorded."""
  return _should_record_summaries_internal(default_state=False)


@tf_export("summary.record_if", v1=[])
@tf_contextlib.contextmanager
def record_if(condition):
  """Sets summary recording on or off per the provided boolean value.

  The provided value can be a python boolean, a scalar boolean Tensor, or
  or a callable providing such a value; if a callable is passed it will be
  invoked on-demand to determine whether summary writing will occur.

  Args:
    condition: can be True, False, a bool Tensor, or a callable providing such.

  Yields:
    Returns a context manager that sets this value on enter and restores the
    previous value on exit.
  """
  old = _summary_state.is_recording
  try:
    _summary_state.is_recording = condition
    yield
  finally:
    _summary_state.is_recording = old


# TODO(apassos) consider how to handle local step here.
def record_summaries_every_n_global_steps(n, global_step=None):
  """Sets the should_record_summaries Tensor to true if global_step % n == 0."""
  if global_step is None:
    global_step = training_util.get_or_create_global_step()
  with ops.device("cpu:0"):
    should = lambda: math_ops.equal(global_step % n, 0)
    if not context.executing_eagerly():
      should = should()
  return record_if(should)


def always_record_summaries():
  """Sets the should_record_summaries Tensor to always true."""
  return record_if(True)


def never_record_summaries():
  """Sets the should_record_summaries Tensor to always false."""
  return record_if(False)


@tf_export("summary.experimental.get_step", v1=[])
def get_step():
  """Returns the default summary step for the current thread.

  Returns:
    The step set by `tf.summary.experimental.set_step()` if one has been set,
    otherwise None.
  """
  return _summary_state.step


@tf_export("summary.experimental.set_step", v1=[])
def set_step(step):
  """Sets the default summary step for the current thread.

  For convenience, this function sets a default value for the `step` parameter
  used in summary-writing functions elsewhere in the API so that it need not
  be explicitly passed in every such invocation. The value can be a constant
  or a variable, and can be retrieved via `tf.summary.experimental.get_step()`.

  Note: when using this with @tf.functions, the step value will be captured at
  the time the function is traced, so changes to the step outside the function
  will not be reflected inside the function unless using a `tf.Variable` step.

  Args:
    step: An `int64`-castable default step value, or None to unset.
  """
  _summary_state.step = step


@tf_export("summary.SummaryWriter", v1=[])
@six.add_metaclass(abc.ABCMeta)
class SummaryWriter(object):
  """Interface representing a stateful summary writer object."""

  @abc.abstractmethod
  def set_as_default(self):
    """Enables this summary writer for the current thread."""
    raise NotImplementedError()

  @abc.abstractmethod
  @tf_contextlib.contextmanager
  def as_default(self):
    """Returns a context manager that enables summary writing."""
    raise NotImplementedError()

  def init(self):
    """Initializes the summary writer."""
    raise NotImplementedError()

  def flush(self):
    """Flushes any buffered data."""
    raise NotImplementedError()

  def close(self):
    """Flushes and closes the summary writer."""
    raise NotImplementedError()


class ResourceSummaryWriter(SummaryWriter):
  """Implementation of SummaryWriter using a SummaryWriterInterface resource."""

  def __init__(self,
               shared_name,
               init_op_fn,
               name=None,
               v2=False,
               metadata=None):
    self._resource = gen_summary_ops.summary_writer(
        shared_name=shared_name, name=name)
    # TODO(nickfelt): cache other constructed ops in graph mode
    self._init_op_fn = init_op_fn
    self._init_op = init_op_fn(self._resource)
    self._v2 = v2
    self._metadata = {} if metadata is None else metadata
    self._closed = False
    if context.executing_eagerly():
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._resource, handle_device="cpu:0")
    else:
      ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME, self._init_op)

  def set_as_default(self):
    """Enables this summary writer for the current thread."""
    if self._v2 and context.executing_eagerly() and self._closed:
      raise RuntimeError("SummaryWriter is already closed")
    _summary_state.writer = self

  @tf_contextlib.contextmanager
  def as_default(self):
    """Returns a context manager that enables summary writing."""
    if self._v2 and context.executing_eagerly() and self._closed:
      raise RuntimeError("SummaryWriter is already closed")
    old = _summary_state.writer
    try:
      _summary_state.writer = self
      yield self
      # Flushes the summary writer in eager mode or in graph functions, but
      # not in legacy graph mode (you're on your own there).
      self.flush()
    finally:
      _summary_state.writer = old

  def init(self):
    """Initializes the summary writer."""
    if self._v2:
      if context.executing_eagerly() and self._closed:
        raise RuntimeError("SummaryWriter is already closed")
      return self._init_op
    # Legacy behavior allows re-initializing the resource.
    return self._init_op_fn(self._resource)

  def flush(self):
    """Flushes any buffered data."""
    if self._v2 and context.executing_eagerly() and self._closed:
      return
    return _flush_fn(writer=self)

  def close(self):
    """Flushes and closes the summary writer."""
    if self._v2 and context.executing_eagerly() and self._closed:
      return
    try:
      with ops.control_dependencies([self.flush()]):
        with ops.device("cpu:0"):
          return gen_summary_ops.close_summary_writer(self._resource)
    finally:
      if self._v2 and context.executing_eagerly():
        self._closed = True


class NoopSummaryWriter(SummaryWriter):
  """A summary writer that does nothing, for create_noop_writer()."""

  def set_as_default(self):
    pass

  @tf_contextlib.contextmanager
  def as_default(self):
    yield

  def init(self):
    pass

  def flush(self):
    pass

  def close(self):
    pass


@tf_export(v1=["summary.initialize"])
def initialize(
    graph=None,  # pylint: disable=redefined-outer-name
    session=None):
  """Initializes summary writing for graph execution mode.

  This operation is a no-op when executing eagerly.

  This helper method provides a higher-level alternative to using
  `tf.contrib.summary.summary_writer_initializer_op` and
  `tf.contrib.summary.graph`.

  Most users will also want to call `tf.compat.v1.train.create_global_step`
  which can happen before or after this function is called.

  Args:
    graph: A `tf.Graph` or `tf.compat.v1.GraphDef` to output to the writer.
      This function will not write the default graph by default. When
      writing to an event log file, the associated step will be zero.
    session: So this method can call `tf.Session.run`. This defaults
      to `tf.compat.v1.get_default_session`.

  Raises:
    RuntimeError: If  the current thread has no default
      `tf.contrib.summary.SummaryWriter`.
    ValueError: If session wasn't passed and no default session.
  """
  if context.executing_eagerly():
    return
  if _summary_state.writer is None:
    raise RuntimeError("No default tf.contrib.summary.SummaryWriter found")
  if session is None:
    session = ops.get_default_session()
    if session is None:
      raise ValueError("session must be passed if no default session exists")
  session.run(summary_writer_initializer_op())
  if graph is not None:
    data = _serialize_graph(graph)
    x = array_ops.placeholder(dtypes.string)
    session.run(_graph(x, 0), feed_dict={x: data})


@tf_export("summary.create_file_writer", v1=[])
def create_file_writer_v2(logdir,
                          max_queue=None,
                          flush_millis=None,
                          filename_suffix=None,
                          name=None):
  """Creates a summary file writer for the given log directory.

  Args:
    logdir: a string specifying the directory in which to write an event file.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: a name for the op that creates the writer.

  Returns:
    A SummaryWriter object.
  """
  if logdir is None:
    raise ValueError("logdir cannot be None")
  inside_function = ops.inside_function()
  with ops.name_scope(name, "create_file_writer") as scope, ops.device("cpu:0"):
    # Run init inside an init_scope() to hoist it out of tf.functions.
    with ops.init_scope():
      if context.executing_eagerly():
        _check_create_file_writer_args(
            inside_function,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix)
      logdir = ops.convert_to_tensor(logdir, dtype=dtypes.string)
      if max_queue is None:
        max_queue = constant_op.constant(10)
      if flush_millis is None:
        flush_millis = constant_op.constant(2 * 60 * 1000)
      if filename_suffix is None:
        filename_suffix = constant_op.constant(".v2")
      # Prepend the PID and a process-local UID to the filename suffix to avoid
      # filename collisions within the machine (the filename already contains
      # the hostname to avoid cross-machine collisions).
      unique_prefix = constant_op.constant(".%s.%s" % (os.getpid(), ops.uid()))
      filename_suffix = unique_prefix + filename_suffix
      # Use a unique shared_name to prevent resource sharing.
      if context.executing_eagerly():
        shared_name = context.shared_name()
      else:
        shared_name = ops.name_from_scope_name(scope)  # pylint: disable=protected-access
      return ResourceSummaryWriter(
          shared_name=shared_name,
          init_op_fn=functools.partial(
              gen_summary_ops.create_summary_file_writer,
              logdir=logdir,
              max_queue=max_queue,
              flush_millis=flush_millis,
              filename_suffix=filename_suffix),
          name=name,
          v2=True,
          metadata={"logdir": logdir})


def create_file_writer(logdir,
                       max_queue=None,
                       flush_millis=None,
                       filename_suffix=None,
                       name=None):
  """Creates a summary file writer in the current context under the given name.

  Args:
    logdir: a string, or None. If a string, creates a summary file writer
     which writes to the directory named by the string. If None, returns
     a mock object which acts like a summary writer but does nothing,
     useful to use as a context manager.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: Shared name for this SummaryWriter resource stored to default
      Graph. Defaults to the provided logdir prefixed with `logdir:`. Note: if a
      summary writer resource with this shared name already exists, the returned
      SummaryWriter wraps that resource and the other arguments have no effect.

  Returns:
    Either a summary writer or an empty object which can be used as a
    summary writer.
  """
  if logdir is None:
    return NoopSummaryWriter()
  logdir = str(logdir)
  with ops.device("cpu:0"):
    if max_queue is None:
      max_queue = constant_op.constant(10)
    if flush_millis is None:
      flush_millis = constant_op.constant(2 * 60 * 1000)
    if filename_suffix is None:
      filename_suffix = constant_op.constant(".v2")
    if name is None:
      name = "logdir:" + logdir
    return ResourceSummaryWriter(
        shared_name=name,
        init_op_fn=functools.partial(
            gen_summary_ops.create_summary_file_writer,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix))


def create_db_writer(db_uri,
                     experiment_name=None,
                     run_name=None,
                     user_name=None,
                     name=None):
  """Creates a summary database writer in the current context.

  This can be used to write tensors from the execution graph directly
  to a database. Only SQLite is supported right now. This function
  will create the schema if it doesn't exist. Entries in the Users,
  Experiments, and Runs tables will be created automatically if they
  don't already exist.

  Args:
    db_uri: For example "file:/tmp/foo.sqlite".
    experiment_name: Defaults to YYYY-MM-DD in local time if None.
      Empty string means the Run will not be associated with an
      Experiment. Can't contain ASCII control characters or <>. Case
      sensitive.
    run_name: Defaults to HH:MM:SS in local time if None. Empty string
      means a Tag will not be associated with any Run. Can't contain
      ASCII control characters or <>. Case sensitive.
    user_name: Defaults to system username if None. Empty means the
      Experiment will not be associated with a User. Must be valid as
      both a DNS label and Linux username.
    name: Shared name for this SummaryWriter resource stored to default
      `tf.Graph`.

  Returns:
    A `tf.summary.SummaryWriter` instance.
  """
  with ops.device("cpu:0"):
    if experiment_name is None:
      experiment_name = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    if run_name is None:
      run_name = time.strftime("%H:%M:%S", time.localtime(time.time()))
    if user_name is None:
      user_name = getpass.getuser()
    experiment_name = _cleanse_string(
        "experiment_name", _EXPERIMENT_NAME_PATTERNS, experiment_name)
    run_name = _cleanse_string("run_name", _RUN_NAME_PATTERNS, run_name)
    user_name = _cleanse_string("user_name", _USER_NAME_PATTERNS, user_name)
    return ResourceSummaryWriter(
        shared_name=name,
        init_op_fn=functools.partial(
            gen_summary_ops.create_summary_db_writer,
            db_uri=db_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            user_name=user_name))


@tf_export("summary.create_noop_writer", v1=[])
def create_noop_writer():
  """Returns a summary writer that does nothing.

  This is useful as a placeholder in code that expects a context manager.
  """
  return NoopSummaryWriter()


def _cleanse_string(name, pattern, value):
  if isinstance(value, six.string_types) and pattern.search(value) is None:
    raise ValueError("%s (%s) must match %s" % (name, value, pattern.pattern))
  return ops.convert_to_tensor(value, dtypes.string)


def _nothing():
  """Convenient else branch for when summaries do not record."""
  return constant_op.constant(False)


@tf_export(v1=["summary.all_v2_summary_ops"])
def all_v2_summary_ops():
  """Returns all V2-style summary ops defined in the current default graph.

  This includes ops from TF 2.0 tf.summary and TF 1.x tf.contrib.summary (except
  for `tf.contrib.summary.graph` and `tf.contrib.summary.import_event`), but
  does *not* include TF 1.x tf.summary ops.

  Returns:
    List of summary ops, or None if called under eager execution.
  """
  if context.executing_eagerly():
    return None
  return ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access


def summary_writer_initializer_op():
  """Graph-mode only. Returns the list of ops to create all summary writers.

  Returns:
    The initializer ops.

  Raises:
    RuntimeError: If in Eager mode.
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "tf.contrib.summary.summary_writer_initializer_op is only "
        "supported in graph mode.")
  return ops.get_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME)


_INVALID_SCOPE_CHARACTERS = re.compile(r"[^-_/.A-Za-z0-9]")


@tf_export("summary.experimental.summary_scope", v1=[])
@tf_contextlib.contextmanager
def summary_scope(name, default_name="summary", values=None):
  """Experimental context manager for use when defining a custom summary op.

  This behaves similarly to `tf.name_scope`, except that it returns a generated
  summary tag in addition to the scope name. The tag is structurally similar to
  the scope name - derived from the user-provided name, prefixed with enclosing
  name scopes if any - but we relax the constraint that it be uniquified, as
  well as the character set limitation (so the user-provided name can contain
  characters not legal for scope names; in the scope name these are removed).

  This makes the summary tag more predictable and consistent for the user.

  For example, to define a new summary op called `my_op`:

  ```python
  def my_op(name, my_value, step):
    with tf.summary.summary_scope(name, "MyOp", [my_value]) as (tag, scope):
      my_value = tf.convert_to_tensor(my_value)
      return tf.summary.write(tag, my_value, step=step)
  ```

  Args:
    name: string name for the summary.
    default_name: Optional; if provided, used as default name of the summary.
    values: Optional; passed as `values` parameter to name_scope.

  Yields:
    A tuple `(tag, scope)` as described above.
  """
  name = name or default_name
  current_scope = ops.get_name_scope()
  tag = current_scope + "/" + name if current_scope else name
  # Strip illegal characters from the scope name, and if that leaves nothing,
  # use None instead so we pick up the default name.
  name = _INVALID_SCOPE_CHARACTERS.sub("", name) or None
  with ops.name_scope(name, default_name, values, skip_on_eager=False) as scope:
    yield tag, scope


@tf_export("summary.write", v1=[])
def write(tag, tensor, step=None, metadata=None, name=None):
  """Writes a generic summary to the default SummaryWriter if one exists.

  This exists primarily to support the definition of type-specific summary ops
  like scalar() and image(), and is not intended for direct use unless defining
  a new type-specific summary op.

  Args:
    tag: string tag used to identify the summary (e.g. in TensorBoard), usually
      generated with `tf.summary.summary_scope`
    tensor: the Tensor holding the summary data to write or a callable that
      returns this Tensor. If a callable is passed, it will only be called when
      a default SummaryWriter exists and the recording condition specified by
      `record_if()` is met.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    metadata: Optional SummaryMetadata, as a proto or serialized bytes
    name: Optional string name for this op.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  with ops.name_scope(name, "write_summary") as scope:
    if _summary_state.writer is None:
      return constant_op.constant(False)
    if step is None:
      step = get_step()
      if step is None:
        raise ValueError("No step set via 'step' argument or "
                         "tf.summary.experimental.set_step()")
    if metadata is None:
      serialized_metadata = b""
    elif hasattr(metadata, "SerializeToString"):
      serialized_metadata = metadata.SerializeToString()
    else:
      serialized_metadata = metadata

    def record():
      """Record the actual summary and return True."""
      # Note the identity to move the tensor to the CPU.
      with ops.device("cpu:0"):
        summary_tensor = tensor() if callable(tensor) else array_ops.identity(
            tensor)
        write_summary_op = gen_summary_ops.write_summary(
            _summary_state.writer._resource,  # pylint: disable=protected-access
            step,
            summary_tensor,
            tag,
            serialized_metadata,
            name=scope)
        with ops.control_dependencies([write_summary_op]):
          return constant_op.constant(True)

    op = smart_cond.smart_cond(
        _should_record_summaries_v2(), record, _nothing, name="summary_cond")
    if not context.executing_eagerly():
      ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
    return op


@tf_export("summary.experimental.write_raw_pb", v1=[])
def write_raw_pb(tensor, step=None, name=None):
  """Writes a summary using raw `tf.compat.v1.Summary` protocol buffers.

  Experimental: this exists to support the usage of V1-style manual summary
  writing (via the construction of a `tf.compat.v1.Summary` protocol buffer)
  with the V2 summary writing API.

  Args:
    tensor: the string Tensor holding one or more serialized `Summary` protobufs
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    name: Optional string name for this op.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  with ops.name_scope(name, "write_raw_pb") as scope:
    if _summary_state.writer is None:
      return constant_op.constant(False)
    if step is None:
      step = get_step()
      if step is None:
        raise ValueError("No step set via 'step' argument or "
                         "tf.summary.experimental.set_step()")

    def record():
      """Record the actual summary and return True."""
      # Note the identity to move the tensor to the CPU.
      with ops.device("cpu:0"):
        raw_summary_op = gen_summary_ops.write_raw_proto_summary(
            _summary_state.writer._resource,  # pylint: disable=protected-access
            step,
            array_ops.identity(tensor),
            name=scope)
        with ops.control_dependencies([raw_summary_op]):
          return constant_op.constant(True)

    with ops.device("cpu:0"):
      op = smart_cond.smart_cond(
          _should_record_summaries_v2(), record, _nothing, name="summary_cond")
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
      return op


def summary_writer_function(name, tensor, function, family=None):
  """Helper function to write summaries.

  Args:
    name: name of the summary
    tensor: main tensor to form the summary
    function: function taking a tag and a scope which writes the summary
    family: optional, the summary's family

  Returns:
    The result of writing the summary.
  """
  name_scope = ops.get_name_scope()
  if name_scope:
    # Add a slash to allow reentering the name scope.
    name_scope += "/"
  def record():
    with ops.name_scope(name_scope), summary_op_util.summary_scope(
        name, family, values=[tensor]) as (tag, scope):
      with ops.control_dependencies([function(tag, scope)]):
        return constant_op.constant(True)

  if _summary_state.writer is None:
    return control_flow_ops.no_op()
  with ops.device("cpu:0"):
    op = smart_cond.smart_cond(
        should_record_summaries(), record, _nothing, name="")
    if not context.executing_eagerly():
      ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
  return op


def generic(name, tensor, metadata=None, family=None, step=None):
  """Writes a tensor summary if possible."""

  def function(tag, scope):
    if metadata is None:
      serialized_metadata = constant_op.constant("")
    elif hasattr(metadata, "SerializeToString"):
      serialized_metadata = constant_op.constant(metadata.SerializeToString())
    else:
      serialized_metadata = metadata
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        array_ops.identity(tensor),
        tag,
        serialized_metadata,
        name=scope)
  return summary_writer_function(name, tensor, function, family=family)


def scalar(name, tensor, family=None, step=None):
  """Writes a scalar summary if possible.

  Unlike `tf.contrib.summary.generic` this op may change the dtype
  depending on the writer, for both practical and efficiency concerns.

  Args:
    name: An arbitrary name for this summary.
    tensor: A `tf.Tensor` Must be one of the following types:
      `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`,
      `int8`, `uint16`, `half`, `uint32`, `uint64`.
    family: Optional, the summary's family.
    step: The `int64` monotonic step variable, which defaults
      to `tf.compat.v1.train.get_global_step`.

  Returns:
    The created `tf.Operation` or a `tf.no_op` if summary writing has
    not been enabled for this context.
  """

  def function(tag, scope):
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_scalar_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def histogram(name, tensor, family=None, step=None):
  """Writes a histogram summary if possible."""

  def function(tag, scope):
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_histogram_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def image(name, tensor, bad_color=None, max_images=3, family=None, step=None):
  """Writes an image summary if possible."""

  def function(tag, scope):
    bad_color_ = (constant_op.constant([255, 0, 0, 255], dtype=dtypes.uint8)
                  if bad_color is None else bad_color)
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_image_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        bad_color_,
        max_images,
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def audio(name, tensor, sample_rate, max_outputs, family=None, step=None):
  """Writes an audio summary if possible."""

  def function(tag, scope):
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_audio_summary(
        _summary_state.writer._resource,  # pylint: disable=protected-access
        _choose_step(step),
        tag,
        array_ops.identity(tensor),
        sample_rate=sample_rate,
        max_outputs=max_outputs,
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def graph(param, step=None, name=None):
  """Writes a TensorFlow graph to the summary interface.

  The graph summary is, strictly speaking, not a summary. Conditions
  like `tf.summary.should_record_summaries` do not apply. Only
  a single graph can be associated with a particular run. If multiple
  graphs are written, then only the last one will be considered by
  TensorBoard.

  When not using eager execution mode, the user should consider passing
  the `graph` parameter to `tf.compat.v1.summary.initialize` instead of
  calling this function. Otherwise special care needs to be taken when
  using the graph to record the graph.

  Args:
    param: A `tf.Tensor` containing a serialized graph proto. When
      eager execution is enabled, this function will automatically
      coerce `tf.Graph`, `tf.compat.v1.GraphDef`, and string types.
    step: The global step variable. This doesn't have useful semantics
      for graph summaries, but is used anyway, due to the structure of
      event log files. This defaults to the global step.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation` or a `tf.no_op` if summary writing has
    not been enabled for this context.

  Raises:
    TypeError: If `param` isn't already a `tf.Tensor` in graph mode.
  """
  if not context.executing_eagerly() and not isinstance(param, ops.Tensor):
    raise TypeError("graph() needs a tf.Tensor (e.g. tf.placeholder) in graph "
                    "mode, but was: %s" % type(param))
  writer = _summary_state.writer
  if writer is None:
    return control_flow_ops.no_op()
  with ops.device("cpu:0"):
    if isinstance(param, (ops.Graph, graph_pb2.GraphDef)):
      tensor = ops.convert_to_tensor(_serialize_graph(param), dtypes.string)
    else:
      tensor = array_ops.identity(param)
    return gen_summary_ops.write_graph_summary(
        writer._resource, _choose_step(step), tensor, name=name)  # pylint: disable=protected-access


_graph = graph  # for functions with a graph parameter


def import_event(tensor, name=None):
  """Writes a `tf.compat.v1.Event` binary proto.

  This can be used to import existing event logs into a new summary writer sink.
  Please note that this is lower level than the other summary functions and
  will ignore the `tf.summary.should_record_summaries` setting.

  Args:
    tensor: A `tf.Tensor` of type `string` containing a serialized
      `tf.compat.v1.Event` proto.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation`.
  """
  return gen_summary_ops.import_event(
      _summary_state.writer._resource, tensor, name=name)  # pylint: disable=protected-access


@tf_export("summary.flush", v1=[])
def flush(writer=None, name=None):
  """Forces summary writer to send any buffered data to storage.

  This operation blocks until that finishes.

  Args:
    writer: The `tf.summary.SummaryWriter` resource to flush.
      The thread default will be used if this parameter is None.
      Otherwise a `tf.no_op` is returned.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation`.
  """
  if writer is None:
    writer = _summary_state.writer
    if writer is None:
      return control_flow_ops.no_op()
  if isinstance(writer, ResourceSummaryWriter):
    resource = writer._resource  # pylint: disable=protected-access
  else:
    # Assume we were passed a raw resource tensor.
    resource = writer
  with ops.device("cpu:0"):
    return gen_summary_ops.flush_summary_writer(resource, name=name)


_flush_fn = flush  # for within SummaryWriter.flush()


def eval_dir(model_dir, name=None):
  """Construct a logdir for an eval summary writer."""
  return os.path.join(model_dir, "eval" if not name else "eval_" + name)


@deprecation.deprecated(date=None,
                        instructions="Renamed to create_file_writer().")
def create_summary_file_writer(*args, **kwargs):
  """Please use `tf.contrib.summary.create_file_writer`."""
  logging.warning("Deprecation Warning: create_summary_file_writer was renamed "
                  "to create_file_writer")
  return create_file_writer(*args, **kwargs)


def _serialize_graph(arbitrary_graph):
  if isinstance(arbitrary_graph, ops.Graph):
    return arbitrary_graph.as_graph_def(add_shapes=True).SerializeToString()
  else:
    return arbitrary_graph.SerializeToString()


def _choose_step(step):
  if step is None:
    return training_util.get_or_create_global_step()
  if not isinstance(step, ops.Tensor):
    return ops.convert_to_tensor(step, dtypes.int64)
  return step


def _check_create_file_writer_args(inside_function, **kwargs):
  """Helper to check the validity of arguments to a create_file_writer() call.

  Args:
    inside_function: whether the create_file_writer() call is in a tf.function
    **kwargs: the arguments to check, as kwargs to give them names.

  Raises:
    ValueError: if the arguments are graph tensors.
  """
  for arg_name, arg in kwargs.items():
    if not isinstance(arg, ops.EagerTensor) and tensor_util.is_tensor(arg):
      if inside_function:
        raise ValueError(
            "Invalid graph Tensor argument \"%s=%s\" to create_file_writer() "
            "inside an @tf.function. The create call will be lifted into the "
            "outer eager execution context, so it cannot consume graph tensors "
            "defined inside the function body." % (arg_name, arg))
      else:
        raise ValueError(
            "Invalid graph Tensor argument \"%s=%s\" to eagerly executed "
            "create_file_writer()." % (arg_name, arg))


def run_metadata(name, data, step=None):
  """Writes entire RunMetadata summary.

  A RunMetadata can contain DeviceStats, partition graphs, and function graphs.
  Please refer to the proto for definition of each field.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A RunMetadata proto to write.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = summary_pb2.SummaryMetadata()
  # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
  # the rationale.
  summary_metadata.plugin_data.plugin_name = "graph_run_metadata"
  # version number = 1
  summary_metadata.plugin_data.content = b"1"

  with summary_scope(name,
                     "graph_run_metadata_summary",
                     [data, step]) as (tag, _):
    with ops.device("cpu:0"):
      tensor = constant_op.constant(data.SerializeToString(),
                                    dtype=dtypes.string)
    return write(
        tag=tag,
        tensor=tensor,
        step=step,
        metadata=summary_metadata)


def run_metadata_graphs(name, data, step=None):
  """Writes graphs from a RunMetadata summary.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A RunMetadata proto to write.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = summary_pb2.SummaryMetadata()
  # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
  # the rationale.
  summary_metadata.plugin_data.plugin_name = "graph_run_metadata_graph"
  # version number = 1
  summary_metadata.plugin_data.content = b"1"

  data = config_pb2.RunMetadata(
      function_graphs=data.function_graphs,
      partition_graphs=data.partition_graphs)

  with summary_scope(name,
                     "graph_run_metadata_graph_summary",
                     [data, step]) as (tag, _):
    with ops.device("cpu:0"):
      tensor = constant_op.constant(data.SerializeToString(),
                                    dtype=dtypes.string)
    return write(
        tag=tag,
        tensor=tensor,
        step=step,
        metadata=summary_metadata)


def keras_model(name, data, step=None):
  """Writes a Keras model as JSON to as a Summary.

  Writing the Keras model configuration allows the TensorBoard graph plugin to
  render a conceptual graph, as opposed to graph of ops. In case the model fails
  to serialize as JSON, it ignores and returns False.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A Keras Model to write.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.

  Returns:
    True on success, or False if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = summary_pb2.SummaryMetadata()
  # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
  # the rationale.
  summary_metadata.plugin_data.plugin_name = "graph_keras_model"
  # version number = 1
  summary_metadata.plugin_data.content = b"1"

  try:
    json_string = data.to_json()
  except Exception as exc:  # pylint: disable=broad-except
    # An exception should not break a model code.
    logging.warn("Model failed to serialize as JSON. Ignoring... %s" % exc)
    return False

  with summary_scope(name, "graph_keras_model", [data, step]) as (tag, _):
    with ops.device("cpu:0"):
      tensor = constant_op.constant(json_string, dtype=dtypes.string)
    return write(
        tag=tag,
        tensor=tensor,
        step=step,
        metadata=summary_metadata)


_TraceContext = collections.namedtuple("TraceContext", ("graph", "profiler"))
_current_trace_context_lock = threading.Lock()
_current_trace_context = None


@tf_export("summary.trace_on", v1=[])
def trace_on(graph=True, profiler=False):  # pylint: disable=redefined-outer-name
  """Starts a trace to record computation graphs and profiling information.

  Must be invoked in eager mode.

  When enabled, TensorFlow runtime will collection information that can later be
  exported and consumed by TensorBoard. The trace is activated across the entire
  TensorFlow runtime and affects all threads of execution.

  To stop the trace and export the collected information, use
  `tf.summary.trace_export`. To stop the trace without exporting, use
  `tf.summary.trace_off`.

  Args:
    graph: If True, enables collection of executed graphs. It includes ones from
        tf.function invocation and ones from the legacy graph mode. The default
        is True.
    profiler: If True, enables the advanced profiler. Enabling profiler
        implicitly enables the graph collection. The profiler may incur a high
        memory overhead. The default is False.

  """
  if ops.inside_function():
    logging.warn("Cannot enable trace inside a tf.function.")
    return
  if not context.executing_eagerly():
    logging.warn("Must enable trace in eager mode.")
    return

  global _current_trace_context
  with _current_trace_context_lock:
    if _current_trace_context:
      logging.warn("Trace already enabled")
      return

    if graph and not profiler:
      context.context().enable_graph_collection()
    if profiler:
      context.context().enable_run_metadata()
      _profiler.start()

    _current_trace_context = _TraceContext(graph=graph, profiler=profiler)


@tf_export("summary.trace_export", v1=[])
def trace_export(name, step=None, profiler_outdir=None):
  """Stops and exports the active trace as a Summary and/or profile file.

  Stops the trace and exports all metadata collected during the trace to the
  default SummaryWriter, if one has been set.

  Args:
    name: A name for the summary to be written.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    profiler_outdir: Output directory for profiler. This is only used when the
      profiler was enabled when the trace was started. In that case, if there is
      a logdir-based default SummaryWriter, this defaults to the same directory,
      but otherwise the argument must be passed.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  global _current_trace_context

  if ops.inside_function():
    logging.warn("Cannot export trace inside a tf.function.")
    return
  if not context.executing_eagerly():
    logging.warn("Can only export trace while executing eagerly.")
    return

  with _current_trace_context_lock:
    if _current_trace_context is None:
      raise ValueError("Must enable trace before export.")
    graph, profiler = _current_trace_context  # pylint: disable=redefined-outer-name
    if profiler_outdir is None \
        and isinstance(_summary_state.writer, ResourceSummaryWriter):
      logdir = _summary_state.writer._metadata.get("logdir")  # pylint: disable=protected-access
      if logdir is not None:
        profiler_outdir = logdir
    if profiler and profiler_outdir is None:
      raise ValueError("Must set profiler_outdir or "
                       "enable summary writer with logdir.")

  run_meta = context.context().export_run_metadata()

  if graph and not profiler:
    run_metadata_graphs(name, run_meta, step)
  else:
    run_metadata(name, run_meta, step)

  if profiler:
    _profiler.save(profiler_outdir, _profiler.stop())

  trace_off()


@tf_export("summary.trace_off", v1=[])
def trace_off():
  """Stops the current trace and discards any collected information."""
  global _current_trace_context
  with _current_trace_context_lock:
    _current_trace_context = None

  # Disabling run_metadata disables graph collection as well.
  context.context().disable_run_metadata()

  # profiler only has start and stop. One needs to stop in order to export
  # and stopping when it is not running will raise an error.
  try:
    _profiler.stop()
  except _profiler.ProfilerNotRunningError:
    pass
