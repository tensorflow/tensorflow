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

import getpass
import os
import re
import time

import six

from tensorflow.contrib.summary import gen_summary_ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.util import tf_contextlib


# Name for a collection which is expected to have at most a single boolean
# Tensor. If this tensor is True the summary ops will record summaries.
_SHOULD_RECORD_SUMMARIES_NAME = "ShouldRecordSummaries"

_SUMMARY_WRITER_INIT_COLLECTION_NAME = "_SUMMARY_WRITER_V2"

_EXPERIMENT_NAME_PATTERNS = re.compile(r"^[^\x00-\x1F<>]{0,256}$")
_RUN_NAME_PATTERNS = re.compile(r"^[^\x00-\x1F<>]{0,512}$")
_USER_NAME_PATTERNS = re.compile(r"^[a-z]([-a-z0-9]{0,29}[a-z0-9])?$", re.I)


def should_record_summaries():
  """Returns boolean Tensor which is true if summaries should be recorded."""
  should_record_collection = ops.get_collection(_SHOULD_RECORD_SUMMARIES_NAME)
  if not should_record_collection:
    return False
  if len(should_record_collection) != 1:
    raise ValueError(
        "More than one tensor specified for whether summaries "
        "should be recorded: %s" % should_record_collection)
  return should_record_collection[0]


# TODO(apassos) consider how to handle local step here.
@tf_contextlib.contextmanager
def record_summaries_every_n_global_steps(n, global_step=None):
  """Sets the should_record_summaries Tensor to true if global_step % n == 0."""
  if global_step is None:
    global_step = training_util.get_or_create_global_step()
  collection_ref = ops.get_collection_ref(_SHOULD_RECORD_SUMMARIES_NAME)
  old = collection_ref[:]
  with ops.device("cpu:0"):
    collection_ref[:] = [math_ops.equal(global_step % n, 0)]
  yield
  collection_ref[:] = old


@tf_contextlib.contextmanager
def always_record_summaries():
  """Sets the should_record_summaries Tensor to always true."""
  collection_ref = ops.get_collection_ref(_SHOULD_RECORD_SUMMARIES_NAME)
  old = collection_ref[:]
  collection_ref[:] = [True]
  yield
  collection_ref[:] = old


@tf_contextlib.contextmanager
def never_record_summaries():
  """Sets the should_record_summaries Tensor to always false."""
  collection_ref = ops.get_collection_ref(_SHOULD_RECORD_SUMMARIES_NAME)
  old = collection_ref[:]
  collection_ref[:] = [False]
  yield
  collection_ref[:] = old


class SummaryWriter(object):
  """Encapsulates a stateful summary writer resource.

  See also:
  - @{tf.contrib.summary.create_file_writer}
  - @{tf.contrib.summary.create_db_writer}
  """

  def  __init__(self, resource):
    self._resource = resource
    if context.executing_eagerly() and self._resource is not None:
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._resource, handle_device="cpu:0")

  def set_as_default(self):
    """Enables this summary writer for the current thread."""
    context.context().summary_writer_resource = self._resource

  @tf_contextlib.contextmanager
  def as_default(self):
    """Enables summary writing within a `with` block."""
    if self._resource is None:
      yield self
    else:
      old = context.context().summary_writer_resource
      context.context().summary_writer_resource = self._resource
      yield self
      # Flushes the summary writer in eager mode or in graph functions, but not
      # in legacy graph mode (you're on your own there).
      with ops.device("cpu:0"):
        gen_summary_ops.flush_summary_writer(self._resource)
      context.context().summary_writer_resource = old


def initialize(
    graph=None,  # pylint: disable=redefined-outer-name
    session=None):
  """Initializes summary writing for graph execution mode.

  This helper method provides a higher-level alternative to using
  @{tf.contrib.summary.summary_writer_initializer_op} and
  @{tf.contrib.summary.graph}.

  Most users will also want to call @{tf.train.create_global_step}
  which can happen before or after this function is called.

  Args:
    graph: A @{tf.Graph} or @{tf.GraphDef} to output to the writer.
      This function will not write the default graph by default. When
      writing to an event log file, the associated step will be zero.
    session: So this method can call @{tf.Session.run}. This defaults
      to @{tf.get_default_session}.

  Raises:
    RuntimeError: If  the current thread has no default
      @{tf.contrib.summary.SummaryWriter}.
    ValueError: If session wasn't passed and no default session.
  """
  if context.executing_eagerly():
    return
  if context.context().summary_writer_resource is None:
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


def create_file_writer(logdir,
                       max_queue=None,
                       flush_millis=None,
                       filename_suffix=None,
                       name=None):
  """Creates a summary file writer in the current context.

  Args:
    logdir: a string, or None. If a string, creates a summary file writer
     which writes to the directory named by the string. If None, returns
     a mock object which acts like a summary writer but does nothing,
     useful to use as a context manager.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this.
    flush_millis: the largest interval between flushes.
    filename_suffix: optional suffix for the event file name.
    name: Shared name for this SummaryWriter resource stored to default
      Graph.

  Returns:
    Either a summary writer or an empty object which can be used as a
    summary writer.
  """
  if logdir is None:
    return SummaryWriter(None)
  with ops.device("cpu:0"):
    if max_queue is None:
      max_queue = constant_op.constant(10)
    if flush_millis is None:
      flush_millis = constant_op.constant(2 * 60 * 1000)
    if filename_suffix is None:
      filename_suffix = constant_op.constant(".v2")
    return _make_summary_writer(
        name,
        gen_summary_ops.create_summary_file_writer,
        logdir=logdir,
        max_queue=max_queue,
        flush_millis=flush_millis,
        filename_suffix=filename_suffix)


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
      @{tf.Graph}.

  Returns:
    A @{tf.contrib.summary.SummaryWriter} instance.
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
    return _make_summary_writer(
        name,
        gen_summary_ops.create_summary_db_writer,
        db_uri=db_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        user_name=user_name)


def _make_summary_writer(name, factory, **kwargs):
  resource = gen_summary_ops.summary_writer(shared_name=name)
  # TODO(apassos): Consider doing this instead.
  # node = factory(resource, **kwargs)
  # if not context.executing_eagerly():
  #   ops.get_default_session().run(node)
  ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME,
                        factory(resource, **kwargs))
  return SummaryWriter(resource)


def _cleanse_string(name, pattern, value):
  if isinstance(value, six.string_types) and pattern.search(value) is None:
    raise ValueError("%s (%s) must match %s" % (name, value, pattern.pattern))
  return ops.convert_to_tensor(value, dtypes.string)


def _nothing():
  """Convenient else branch for when summaries do not record."""
  return constant_op.constant(False)


def all_summary_ops():
  """Graph-mode only. Returns all summary ops.

  Please note this excludes @{tf.contrib.summary.graph} ops.

  Returns:
    The summary ops.
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
  def record():
    with summary_op_util.summary_scope(
        name, family, values=[tensor]) as (tag, scope):
      with ops.control_dependencies([function(tag, scope)]):
        return constant_op.constant(True)

  if context.context().summary_writer_resource is None:
    return control_flow_ops.no_op()
  with ops.device("cpu:0"):
    op = utils.smart_cond(
        should_record_summaries(), record, _nothing, name="")
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
        context.context().summary_writer_resource,
        _choose_step(step),
        array_ops.identity(tensor),
        tag,
        serialized_metadata,
        name=scope)
  return summary_writer_function(name, tensor, function, family=family)


def scalar(name, tensor, family=None, step=None):
  """Writes a scalar summary if possible.

  Unlike @{tf.contrib.summary.generic} this op may change the dtype
  depending on the writer, for both practical and efficiency concerns.

  Args:
    name: An arbitrary name for this summary.
    tensor: A @{tf.Tensor} Must be one of the following types:
      `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`,
      `int8`, `uint16`, `half`, `uint32`, `uint64`.
    family: Optional, the summary's family.
    step: The `int64` monotonic step variable, which defaults
      to @{tf.train.get_global_step}.

  Returns:
    The created @{tf.Operation} or a @{tf.no_op} if summary writing has
    not been enabled for this context.
  """

  def function(tag, scope):
    # Note the identity to move the tensor to the CPU.
    return gen_summary_ops.write_scalar_summary(
        context.context().summary_writer_resource,
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
        context.context().summary_writer_resource,
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
        context.context().summary_writer_resource,
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
        context.context().summary_writer_resource,
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
  like @{tf.contrib.summary.never_record_summaries} do not apply. Only
  a single graph can be associated with a particular run. If multiple
  graphs are written, then only the last one will be considered by
  TensorBoard.

  When not using eager execution mode, the user should consider passing
  the `graph` parameter to @{tf.contrib.summary.initialize} instead of
  calling this function. Otherwise special care needs to be taken when
  using the graph to record the graph.

  Args:
    param: A @{tf.Tensor} containing a serialized graph proto. When
      eager execution is enabled, this function will automatically
      coerce @{tf.Graph}, @{tf.GraphDef}, and string types.
    step: The global step variable. This doesn't have useful semantics
      for graph summaries, but is used anyway, due to the structure of
      event log files. This defaults to the global step.
    name: A name for the operation (optional).

  Returns:
    The created @{tf.Operation} or a @{tf.no_op} if summary writing has
    not been enabled for this context.

  Raises:
    TypeError: If `param` isn't already a @{tf.Tensor} in graph mode.
  """
  if not context.executing_eagerly() and not isinstance(param, ops.Tensor):
    raise TypeError("graph() needs a tf.Tensor (e.g. tf.placeholder) in graph "
                    "mode, but was: %s" % type(param))
  writer = context.context().summary_writer_resource
  if writer is None:
    return control_flow_ops.no_op()
  with ops.device("cpu:0"):
    if isinstance(param, (ops.Graph, graph_pb2.GraphDef)):
      tensor = ops.convert_to_tensor(_serialize_graph(param), dtypes.string)
    else:
      tensor = array_ops.identity(param)
    return gen_summary_ops.write_graph_summary(
        writer, _choose_step(step), tensor, name=name)


_graph = graph  # for functions with a graph parameter


def import_event(tensor, name=None):
  """Writes a @{tf.Event} binary proto.

  When using create_db_writer(), this can be used alongside
  @{tf.TFRecordReader} to load event logs into the database. Please
  note that this is lower level than the other summary functions and
  will ignore any conditions set by methods like
  @{tf.contrib.summary.should_record_summaries}.

  Args:
    tensor: A @{tf.Tensor} of type `string` containing a serialized
      @{tf.Event} proto.
    name: A name for the operation (optional).

  Returns:
    The created @{tf.Operation}.
  """
  return gen_summary_ops.import_event(
      context.context().summary_writer_resource, tensor, name=name)


def flush(writer=None, name=None):
  """Forces summary writer to send any buffered data to storage.

  This operation blocks until that finishes.

  Args:
    writer: The @{tf.contrib.summary.SummaryWriter} resource to flush.
      The thread default will be used if this parameter is None.
      Otherwise a @{tf.no_op} is returned.
    name: A name for the operation (optional).

  Returns:
    The created @{tf.Operation}.
  """
  if writer is None:
    writer = context.context().summary_writer_resource
    if writer is None:
      return control_flow_ops.no_op()
  return gen_summary_ops.flush_summary_writer(writer, name=name)


def eval_dir(model_dir, name=None):
  """Construct a logdir for an eval summary writer."""
  return os.path.join(model_dir, "eval" if not name else "eval_" + name)


def create_summary_file_writer(*args, **kwargs):
  """Please use @{tf.contrib.summary.create_file_writer}."""
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
