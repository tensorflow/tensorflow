# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A client interface for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import threading

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


class SessionInterface(object):
  """Base class for implementations of TensorFlow client sessions."""

  @property
  def graph(self):
    """The underlying TensorFlow graph, to be used in building Operations."""
    raise NotImplementedError('graph')

  @property
  def sess_str(self):
    """The TensorFlow process to which this session will connect."""
    raise NotImplementedError('sess_str')

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Runs operations in the session. See `Session.run()` for details."""
    raise NotImplementedError('run')

  def partial_run_setup(self, fetches, feeds=None):
    """Sets up the feeds and fetches for partial runs in the session."""
    raise NotImplementedError('partial_run_setup')

  def partial_run(self, handle, fetches, feed_dict=None):
    """Continues the execution with additional feeds and fetches."""
    raise NotImplementedError('partial_run')

def _get_indexed_slices_value_from_fetches(fetched_vals):
  return ops.IndexedSlicesValue(fetched_vals[0], fetched_vals[1],
                                fetched_vals[2]
                                if len(fetched_vals) == 3 else None)


def _get_feeds_for_indexed_slices(feed, feed_val):
  return list(zip([feed.values, feed.indices] if feed.dense_shape is None else
                  [feed.values, feed.indices, feed.dense_shape], feed_val))


class BaseSession(SessionInterface):
  """A class for interacting with a TensorFlow computation.

  The BaseSession enables incremental graph building with inline
  execution of Operations and evaluation of Tensors.
  """

  def __init__(self, target='', graph=None, config=None):
    """Constructs a new TensorFlow session.

    Args:
      target: (Optional) The TensorFlow execution engine to connect to.
      graph: (Optional) The graph to be used. If this argument is None,
        the default graph will be used.
      config: (Optional) ConfigProto proto used to configure the session.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow session.
    """
    if graph is None:
      self._graph = ops.get_default_graph()
    else:
      self._graph = graph

    self._opened = False
    self._closed = False

    self._current_version = 0
    self._extend_lock = threading.Lock()
    self._target = target

    self._delete_lock = threading.Lock()
    self._dead_handles = []

    self._session = None
    self._config = config
    self._add_shapes = config.graph_options.infer_shapes if (
        config and config.graph_options) else False

    try:
      opts = tf_session.TF_NewSessionOptions(target=target, config=config)
      with errors.raise_exception_on_not_ok_status() as status:
        self._session = tf_session.TF_NewSession(opts, status)
    finally:
      tf_session.TF_DeleteSessionOptions(opts)

  def close(self):
    """Closes this session.

    Calling this method frees all resources associated with the session.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        closing the TensorFlow session.
    """
    with self._extend_lock:
      if self._opened and not self._closed:
        self._closed = True
        with errors.raise_exception_on_not_ok_status() as status:
          tf_session.TF_CloseSession(self._session, status)

  def __del__(self):
    self.close()
    if self._session is not None:
      with errors.raise_exception_on_not_ok_status() as status:
        tf_session.TF_DeleteSession(self._session, status)
      self._session = None

  @property
  def graph(self):
    """The graph that was launched in this session."""
    return self._graph

  @property
  def graph_def(self):
    """A serializable version of the underlying TensorFlow graph.

    Returns:
      A graph_pb2.GraphDef proto containing nodes for all of the Operations in
      the underlying TensorFlow graph.
    """
    return self._graph.as_graph_def(add_shapes=self._add_shapes)

  @property
  def sess_str(self):
    return self._target

  def as_default(self):
    """Returns a context manager that makes this object the default session.

    Use with the `with` keyword to specify that calls to
    [`Operation.run()`](../../api_docs/python/framework.md#Operation.run) or
    [`Tensor.run()`](../../api_docs/python/framework.md#Tensor.run) should be
    executed in this session.

    ```python
    c = tf.constant(..)
    sess = tf.Session()

    with sess.as_default():
      assert tf.get_default_session() is sess
      print(c.eval())
    ```

    To get the current default session, use
    [`tf.get_default_session()`](#get_default_session).


    *N.B.* The `as_default` context manager *does not* close the
    session when you exit the context, and you must close the session
    explicitly.

    ```python
    c = tf.constant(...)
    sess = tf.Session()
    with sess.as_default():
      print(c.eval())
    # ...
    with sess.as_default():
      print(c.eval())

    sess.close()
    ```

    Alternatively, you can use `with tf.Session():` to create a
    session that is automatically closed on exiting the context,
    including when an uncaught exception is raised.

    *N.B.* The default graph is a property of the current thread. If you
    create a new thread, and wish to use the default session in that
    thread, you must explicitly add a `with sess.as_default():` in that
    thread's function.

    Returns:
      A context manager using this session as the default session.

    """
    return ops.default_session(self)

  # Eventually, this registration could be opened up to support custom
  # Tensor expansions. Expects tuples of (Type, fetch_fn, feed_fn1, feed_fn2),
  # where the signatures are:
  #   fetch_fn : Type -> (list of Tensors,
  #                       lambda: list of fetched np.ndarray -> TypeVal)
  #   feed_fn1 : Type, TypeVal -> list of (Tensor, value)
  #   feed_fn2 : Type -> list of Tensors
  # Conceptually, fetch_fn describes how to expand fetch into its
  # component Tensors and how to contracting the fetched results back into
  # a single return value. feed_fn describes how to unpack a single fed
  # value and map it to feeds of a Tensor and its corresponding value.
  # pylint: disable=g-long-lambda
  _REGISTERED_EXPANSIONS = [
      # SparseTensors are fetched as SparseTensorValues. They can be fed
      # SparseTensorValues or normal tuples.
      (ops.SparseTensor,
       lambda fetch: (
           [fetch.indices, fetch.values, fetch.shape],
           lambda fetched_vals: ops.SparseTensorValue(*fetched_vals)),
       lambda feed, feed_val: list(zip(
           [feed.indices, feed.values, feed.shape], feed_val)),
       lambda feed: [feed.indices, feed.values, feed.shape]),
      # IndexedSlices are fetched as IndexedSlicesValues. They can be fed
      # IndexedSlicesValues or normal tuples.
      (ops.IndexedSlices,
       lambda fetch: (
           [fetch.values, fetch.indices] if fetch.dense_shape is None
           else [fetch.values, fetch.indices, fetch.dense_shape],
           _get_indexed_slices_value_from_fetches),
       _get_feeds_for_indexed_slices,
       lambda feed: [feed.values, feed.indices] if feed.dense_shape is None
                    else [feed.values, feed.indices, feed.dense_shape]),
      # The default catches all types and performs no expansions.
      (object,
       lambda fetch: ([fetch], lambda fetched_vals: fetched_vals[0]),
       lambda feed, feed_val: [(feed, feed_val)],
       lambda feed: [feed])]
  # pylint: enable=g-long-lambda

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Runs the operations and evaluates the tensors in `fetches`.

    This method runs one "step" of TensorFlow computation, by
    running the necessary graph fragment to execute every `Operation`
    and evaluate every `Tensor` in `fetches`, substituting the values in
    `feed_dict` for the corresponding input values.

    The `fetches` argument may be a list of graph elements or a single
    graph element, and these determine the return value of this
    method. A graph element can be one of the following types:

    * If the *i*th element of `fetches` is an
      [`Operation`](../../api_docs/python/framework.md#Operation), the *i*th
      return value will be `None`.
    * If the *i*th element of `fetches` is a
      [`Tensor`](../../api_docs/python/framework.md#Tensor), the *i*th return
      value will be a numpy ndarray containing the value of that tensor.
    * If the *i*th element of `fetches` is a
      [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
      the *i*th return value will be a
      [`SparseTensorValue`](../../api_docs/python/sparse_ops.md#SparseTensorValue)
      containing the value of that sparse tensor.
    * If the *i*th element of `fetches` is produced by a `get_tensor_handle` op,
      the *i*th return value will be a numpy ndarray containing the handle of
      that tensor.

    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. Each key in `feed_dict` can be
    one of the following types:

    * If the key is a [`Tensor`](../../api_docs/python/framework.md#Tensor), the
      value may be a Python scalar, string, list, or numpy ndarray
      that can be converted to the same `dtype` as that
      tensor. Additionally, if the key is a
      [placeholder](../../api_docs/python/io_ops.md#placeholder), the shape of
      the value will be checked for compatibility with the placeholder.
    * If the key is a
      [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
      the value should be a
      [`SparseTensorValue`](../../api_docs/python/sparse_ops.md#SparseTensorValue).

    Each value in `feed_dict` must be convertible to a numpy array of the dtype
    of the corresponding key.

    The optional `options` argument expects a [`RunOptions`] proto. The options
    allow controlling the behavior of this particular step (e.g. turning tracing
    on).

    The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
    appropriate, the non-Tensor output of this step will be collected there. For
    example, when users turn on tracing in `options`, the profiled info will be
    collected into this argument and passed back.

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      feed_dict: A dictionary that maps graph elements to values
        (described above).
      options: A [`RunOptions`] protocol buffer
      run_metadata: A [`RunMetadata`] protocol buffer

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list (described above).

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      ValueError: If `fetches` or `feed_dict` keys are invalid or refer to a
        `Tensor` that doesn't exist.
    """
    run_metadata_ptr = tf_session.TF_NewBuffer()
    if options:
      options_ptr = tf_session.TF_NewBufferFromString(
          compat.as_bytes(options.SerializeToString()))
    else:
      options_ptr = None

    try:
      result = self._run(None, fetches, feed_dict, options_ptr,
                         run_metadata_ptr)
      if run_metadata:
        proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
        run_metadata.ParseFromString(compat.as_bytes(proto_data))
    finally:
      tf_session.TF_DeleteBuffer(run_metadata_ptr)
      if options:
        tf_session.TF_DeleteBuffer(options_ptr)
    return result

  def partial_run(self, handle, fetches, feed_dict=None):
    """Continues the execution with more feeds and fetches.

    This is EXPERIMENTAL and subject to change.

    To use partial execution, a user first calls `partial_run_setup()` and
    then a sequence of `partial_run()`. `partial_run_setup` specifies the
    list of feeds and fetches that will be used in the subsequent
    `partial_run` calls.

    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. See run() for more information.

    Below is a simple example:

    ```python
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.mul(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    res = sess.partial_run(h, r2, feed_dict={c: res})
    ```

    Args:
      handle: A handle for a sequence of partial runs.
      fetches: A single graph element, or a list of graph elements
        (described above).
      feed_dict: A dictionary that maps graph elements to values
        (described above).

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list (described above).

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
    return self._run(handle, fetches, feed_dict, None, None)

  def partial_run_setup(self, fetches, feeds=None):
    """Sets up a graph with feeds and fetches for partial run.

    This is EXPERIMENTAL and subject to change.

    Note that contrary to `run`, `feeds` only specifies the graph elements.
    The tensors will be supplied by the subsequent `partial_run` calls.

    Args:
      fetches: A single graph element, or a list of graph elements.
      feeds: A single graph element, or a list of graph elements.

    Returns:
      A handle for partial run.

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      tf.errors.OpError: Or one of its subclasses if a TensorFlow error happens.
    """
    def _feed_fn(feed):
      for tensor_type, _, _, feed_fn in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed)
      raise TypeError('Feed argument %r has invalid type %r'
                      % (feed, type(feed)))

    # Check session.
    if self._closed:
      raise RuntimeError('Attempted to use a closed Session.')
    if self.graph.version == 0:
      raise RuntimeError('The Session graph is empty.  Add operations to the '
                         'graph before calling run().')

    # Validate and process fetches.
    unique_fetches, target_list, _, _ = self._process_fetches(fetches)

    # Create request.
    feed_list = []

    # Validate and process feed_list.
    is_list_feed = isinstance(feeds, (list, tuple))
    if not is_list_feed:
      feeds = [feeds]
    for feed in feeds:
      for subfeed in _feed_fn(feed):
        try:
          subfeed_t = self.graph.as_graph_element(subfeed, allow_tensor=True,
                                                  allow_operation=False)
          feed_list.append(compat.as_bytes(subfeed_t.name))
        except Exception as e:
          e.message = ('Cannot interpret feed_list key as Tensor: '
                       + e.message)
          e.args = (e.message,)
          raise e

    # Set up a graph with feeds and fetches for partial run.
    def _setup_fn(session, feed_list, fetch_list, target_list):
      self._extend_graph()
      with errors.raise_exception_on_not_ok_status() as status:
        return tf_session.TF_PRunSetup(session, feed_list, fetch_list,
                                       target_list, status)

    return self._do_call(_setup_fn, self._session, feed_list, unique_fetches,
                         target_list)

  def _process_fetches(self, fetches):
    """Validate and process fetches."""
    def _fetch_fn(fetch):
      for tensor_type, fetch_fn, _, _ in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(fetch, tensor_type):
          return fetch_fn(fetch)
      raise TypeError('Fetch argument %r has invalid type %r'
                      % (fetch, type(fetch)))

    # Validate and process fetches.
    is_list_fetch = isinstance(fetches, (list, tuple))
    if not is_list_fetch:
      fetches = [fetches]

    unique_fetch_targets = set()
    unique_fetch_handles = {}
    target_list = []

    fetch_info = []
    for fetch in fetches:
      subfetches, fetch_contraction_fn = _fetch_fn(fetch)
      subfetch_names = []
      for subfetch in subfetches:
        try:
          fetch_t = self.graph.as_graph_element(subfetch, allow_tensor=True,
                                                allow_operation=True)
          fetch_name = compat.as_bytes(fetch_t.name)
          if isinstance(fetch_t, ops.Operation):
            target_list.append(fetch_name)
          else:
            subfetch_names.append(fetch_name)
          # Remember the fetch if it is for a tensor handle.
          if (isinstance(fetch_t, ops.Tensor) and
              fetch_t.op.type == 'GetSessionHandle'):
            unique_fetch_handles[fetch_name] = fetch_t.op.inputs[0].dtype
        except TypeError as e:
          raise TypeError('Fetch argument %r of %r has invalid type %r, '
                          'must be a string or Tensor. (%s)'
                          % (subfetch, fetch, type(subfetch), str(e)))
        except ValueError as e:
          raise ValueError('Fetch argument %r of %r cannot be interpreted as a '
                           'Tensor. (%s)' % (subfetch, fetch, str(e)))
        except KeyError as e:
          raise ValueError('Fetch argument %r of %r cannot be interpreted as a '
                           'Tensor. (%s)' % (subfetch, fetch, str(e)))
      unique_fetch_targets.update(subfetch_names)
      fetch_info.append((subfetch_names, fetch_contraction_fn))

    unique_fetch_targets = list(unique_fetch_targets)
    return unique_fetch_targets, target_list, fetch_info, unique_fetch_handles

  def _run(self, handle, fetches, feed_dict, options, run_metadata):
    """Perform either run or partial_run, depending the exitence of `handle`."""
    def _feed_fn(feed, feed_val):
      for tensor_type, _, feed_fn, _ in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed, feed_val)
      raise TypeError('Feed argument %r has invalid type %r'
                      % (feed, type(feed)))

    # Check session.
    if self._closed:
      raise RuntimeError('Attempted to use a closed Session.')
    if self.graph.version == 0:
      raise RuntimeError('The Session graph is empty.  Add operations to the '
                         'graph before calling run().')

    # Validate and process fetches.
    processed_fetches = self._process_fetches(fetches)
    unique_fetches = processed_fetches[0]
    target_list = processed_fetches[1]
    fetch_info = processed_fetches[2]
    unique_handles = processed_fetches[3]

    # Create request.
    feed_dict_string = {}
    feed_map = {}

    # Validate and process feed_dict.
    if feed_dict:
      for feed, feed_val in feed_dict.items():
        for subfeed, subfeed_val in _feed_fn(feed, feed_val):
          try:
            subfeed_t = self.graph.as_graph_element(subfeed, allow_tensor=True,
                                                    allow_operation=False)
          except Exception as e:
            raise TypeError('Cannot interpret feed_dict key as Tensor: '
                            + e.args[0])

          if isinstance(subfeed_val, ops.Tensor):
            raise TypeError('The value of a feed cannot be a tf.Tensor object. '
                            'Acceptable feed values include Python scalars, '
                            'strings, lists, or numpy ndarrays.')

          subfeed_dtype = subfeed_t.dtype.as_numpy_dtype
          if isinstance(subfeed_val,
                        int) and subfeed_dtype(subfeed_val) != subfeed_val:
            raise TypeError(
                'Type of feed value ' + str(subfeed_val) + ' is not'
                ' compatible with Tensor type ' + str(subfeed_dtype) + '.'
                ' Try explicitly setting the type of the feed tensor'
                ' to a larger type (e.g. int64).')

          np_val = np.array(subfeed_val, dtype=subfeed_dtype)

          if not subfeed_t.get_shape().is_compatible_with(np_val.shape):
            raise ValueError(
                'Cannot feed value of shape %r for Tensor %r, '
                'which has shape %r'
                % (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
          if not self.graph.is_feedable(subfeed_t):
            raise ValueError('Tensor %s may not be fed.' % subfeed_t)
          subfeed_name = compat.as_bytes(subfeed_t.name)
          feed_dict_string[subfeed_name] = np_val
          feed_map[subfeed_name] = (subfeed_t, subfeed_val)

    # Run request and get response.
    movers = self._update_with_movers(feed_dict_string, feed_map)
    try:
      results = self._do_run(handle, target_list, unique_fetches,
                             feed_dict_string, options, run_metadata)
    finally:
      # The movers are no longer used. Delete them.
      for handle in movers:
        self._register_dead_handle(handle)

    # User may have fetched the same tensor multiple times, but we
    # only fetch them from the runtime once.  Furthermore, they may
    # be wrapped as a tuple of tensors.  Here we map the results back
    # to what the client asked for.
    # TODO(yuanbyu): Use the contraction_fn in _REGISTERED_EXPANSIONS.
    fetched_results = {}
    for fetch, result in zip(unique_fetches, results):
      dtype = unique_handles.get(fetch)
      if dtype:
        result = session_ops.TensorHandle(result, dtype, self)
      fetched_results[fetch] = result
    ret = []
    for fetch_names, fetch_contraction_fn in fetch_info:
      if fetch_names:
        fetched_vals = [fetched_results[name] for name in fetch_names]
        ret.append(fetch_contraction_fn(fetched_vals))
      else:
        ret.append(None)

    if isinstance(fetches, (list, tuple)):
      return ret
    else:
      return ret[0]

  # Captures the name of a node in an error status.
  _NODEDEF_NAME_RE = re.compile(r'\[\[Node: ([^ ]*?) =')

  def _do_run(self, handle, target_list, fetch_list, feed_dict,
              options, run_metadata):
    """Runs a step based on the given fetches and feeds.

    Args:
      handle: a handle for partial_run. None if this is just a call to run().
      target_list: A list of byte arrays corresponding to names of tensors
        or operations to be run to, but not fetched.
      fetch_list: A list of byte arrays corresponding to names of tensors to
        be fetched and operations to be run.
      feed_dict: A dictionary that maps tensor names (as byte arrays) to
        numpy ndarrays.
      options: A (pointer to a) [`RunOptions`] protocol buffer, or None
      run_metadata: A (pointer to a) [`RunMetadata`] protocol buffer, or None

    Returns:
      A list of numpy ndarrays, corresponding to the elements of
      `fetch_list`.  If the ith element of `fetch_list` contains the
      name of an operation, the first Tensor output of that operation
      will be returned for that element.

    Raises:
      tf.errors.OpError: Or one of its subclasses on error.
    """
    def _run_fn(session, feed_dict, fetch_list, target_list, options,
                run_metadata):
      # Ensure any changes to the graph are reflected in the runtime.
      self._extend_graph()
      with errors.raise_exception_on_not_ok_status() as status:
        if options:
          return tf_session.TF_Run(session, options,
                                   feed_dict, fetch_list, target_list,
                                   status, run_metadata)
        else:
          return tf_session.TF_Run(
              session, None, feed_dict, fetch_list, target_list, status,
              None)

    def _prun_fn(session, handle, feed_dict, fetch_list):
      if target_list:
        raise RuntimeError('partial_run() requires empty target_list.')
      with errors.raise_exception_on_not_ok_status() as status:
        return tf_session.TF_PRun(session, handle, feed_dict, fetch_list,
                                  status)

    if handle is None:
      return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
                           target_list, options, run_metadata)
    else:
      return self._do_call(_prun_fn, self._session, handle, feed_dict,
                           fetch_list)

  def _do_call(self, fn, *args):
    try:
      return fn(*args)
    except errors.OpError as e:
      message = compat.as_text(e.message)
      m = BaseSession._NODEDEF_NAME_RE.search(message)
      node_def = None
      op = None
      if m is not None:
        node_name = m.group(1)
        try:
          op = self._graph.get_operation_by_name(node_name)
          node_def = op.node_def
        except KeyError:
          pass
      raise type(e)(node_def, op, message)

  def _extend_graph(self):
    # Ensure any changes to the graph are reflected in the runtime.
    with self._extend_lock:
      if self._graph.version > self._current_version:
        graph_def = self._graph.as_graph_def(
            from_version=self._current_version,
            add_shapes=self._add_shapes)

        with errors.raise_exception_on_not_ok_status() as status:
          tf_session.TF_ExtendGraph(
              self._session, graph_def.SerializeToString(), status)
        self._opened = True

        self._current_version = self._graph.version

  # The threshold to run garbage collection to delete dead tensors.
  _DEAD_HANDLES_THRESHOLD = 10

  def _register_dead_handle(self, handle):
    # Register a dead handle in the session. Delete the dead tensors when
    # the number of dead tensors exceeds certain threshold.
    tensors_to_delete = None
    with self._delete_lock:
      self._dead_handles.append(handle)
      if len(self._dead_handles) == BaseSession._DEAD_HANDLES_THRESHOLD:
        tensors_to_delete = self._dead_handles
        self._dead_handles = []
    # Delete the dead tensors.
    # TODO(yuanbyu): For now we use a sequence of runs to minimize the graph
    # size and the overhead of graph construction/partitioning.
    if tensors_to_delete:
      for tensor_handle in tensors_to_delete:
        feeds = {}
        fetches = []
        holder, deleter = session_ops._get_handle_deleter(self.graph,
                                                          tensor_handle)
        feeds[holder] = tensor_handle
        fetches.append(deleter)
        self.run(fetches, feed_dict=feeds)

  def _update_with_movers(self, feed_dict, feed_map):
    # If a tensor handle that is fed to a device incompatible placeholder,
    # we move the tensor to the right device, generate a new tensor handle,
    # and update `feed_dict` to use the new handle.
    handle_movers = []
    for feed_name, val in feed_map.items():
      mover = session_ops._get_handle_mover(self.graph, *val)
      if mover:
        handle_movers.append((feed_name, val[1], mover))
    # Transfer a tensor to the right device if needed.
    if not handle_movers:
      return []
    else:
      feeds = {}
      fetches = []
      for _, handle, mover in handle_movers:
        feeds[mover[0]] = handle
        fetches.append(mover[1])
      handles = self.run(fetches, feed_dict=feeds)
      for handle_mover, handle in zip(handle_movers, handles):
        np_val = np.array(handle.handle, dtype=np.object)
        feed_dict[handle_mover[0]] = np_val
      return handles


class Session(BaseSession):
  """A class for running TensorFlow operations.

  A `Session` object encapsulates the environment in which `Operation`
  objects are executed, and `Tensor` objects are evaluated. For
  example:

  ```python
  # Build a graph.
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b

  # Launch the graph in a session.
  sess = tf.Session()

  # Evaluate the tensor `c`.
  print(sess.run(c))
  ```

  A session may own resources, such as
  [variables](../../api_docs/python/state_ops.md#Variable), [queues](../../api_docs/python/io_ops.md#QueueBase),
  and [readers](../../api_docs/python/io_ops.md#ReaderBase). It is important to release
  these resources when they are no longer required. To do this, either
  invoke the [`close()`](#Session.close) method on the session, or use
  the session as a context manager. The following two examples are
  equivalent:

  ```python
  # Using the `close()` method.
  sess = tf.Session()
  sess.run(...)
  sess.close()

  # Using the context manager.
  with tf.Session() as sess:
    sess.run(...)
  ```

  The [`ConfigProto`]
  (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
  protocol buffer exposes various configuration options for a
  session. For example, to create a session that uses soft constraints
  for device placement, and log the resulting placement decisions,
  create a session as follows:

  ```python
  # Launch the graph in a session that allows soft device placement and
  # logs the placement decisions.
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True))
  ```

  @@__init__
  @@run
  @@close

  @@graph

  @@as_default

  """

  def __init__(self, target='', graph=None, config=None):
    """Creates a new TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
    using more than one graph (created with `tf.Graph()` in the same
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
      target: (Optional.) The execution engine to connect to.
        Defaults to using an in-process engine. At present, no value
        other than the empty string is supported.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional.) A [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
        protocol buffer with configuration options for the session.

    """
    super(Session, self).__init__(target, graph, config=config)
    self._context_managers = [self.graph.as_default(), self.as_default()]

  def __enter__(self):
    for context_manager in self._context_managers:
      context_manager.__enter__()
    return self

  def __exit__(self, exec_type, exec_value, exec_tb):
    if exec_type is errors.OpError:
      logging.error('Session closing due to OpError: %s', (exec_value,))

    for context_manager in reversed(self._context_managers):
      context_manager.__exit__(exec_type, exec_value, exec_tb)

    self.close()


class InteractiveSession(BaseSession):
  """A TensorFlow `Session` for use in interactive contexts, such as a shell.

  The only difference with a regular `Session` is that an `InteractiveSession`
  installs itself as the default session on construction.
  The methods [`Tensor.eval()`](../../api_docs/python/framework.md#Tensor.eval)
  and [`Operation.run()`](../../api_docs/python/framework.md#Operation.run)
  will use that session to run ops.

  This is convenient in interactive shells and [IPython
  notebooks](http://ipython.org), as it avoids having to pass an explicit
  `Session` object to run ops.

  For example:

  ```python
  sess = tf.InteractiveSession()
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  # We can just use 'c.eval()' without passing 'sess'
  print(c.eval())
  sess.close()
  ```

  Note that a regular session installs itself as the default session when it
  is created in a `with` statement.  The common usage in non-interactive
  programs is to follow that pattern:

  ```python
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  with tf.Session():
    # We can also use 'c.eval()' here.
    print(c.eval())
  ```

  @@__init__
  @@close
  """

  def __init__(self, target='', graph=None, config=None):
    """Creates a new interactive TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
    using more than one graph (created with `tf.Graph()` in the same
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
      target: (Optional.) The execution engine to connect to.
        Defaults to using an in-process engine. At present, no value
        other than the empty string is supported.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional) `ConfigProto` proto used to configure the session.
    """
    if not config:
      config = config_pb2.ConfigProto()
    # Interactive sessions always place pruned graphs.
    config.graph_options.place_pruned_graph = True

    super(InteractiveSession, self).__init__(target, graph, config)
    self._default_session = self.as_default()
    self._default_session.__enter__()
    self._explicit_graph = graph
    if self._explicit_graph is not None:
      self._default_graph = graph.as_default()
      self._default_graph.__enter__()

  def close(self):
    """Closes an `InteractiveSession`."""
    super(InteractiveSession, self).close()
    if self._explicit_graph is not None:
      self._default_graph.__exit__(None, None, None)
    self._default_session.__exit__(None, None, None)
