"""A client interface for TensorFlow."""

import re
import sys
import threading

import tensorflow.python.platform

import numpy as np

from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import logging


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

  def run(self, fetches, feed_dict=None):
    """Runs operations in the session. See `Session.run()` for details."""
    raise NotImplementedError('Run')


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
      RuntimeError: If an error occurs while creating the TensorFlow
        session.
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

    self._session = None

    try:
      opts = tf_session.TF_NewSessionOptions(target=target, config=config)
      status = tf_session.TF_NewStatus()
      self._session = tf_session.TF_NewSession(opts, status)
      if tf_session.TF_GetCode(status) != 0:
        message = tf_session.TF_Message(status)
        raise RuntimeError(message)

    finally:
      tf_session.TF_DeleteSessionOptions(opts)
      tf_session.TF_DeleteStatus(status)

  def close(self):
    """Closes this session.

    Calling this method frees all resources associated with the session.

    Raises:
      RuntimeError: If an error occurs while closing the session.
    """
    with self._extend_lock:
      if self._opened and not self._closed:
        self._closed = True
        try:
          status = tf_session.TF_NewStatus()
          tf_session.TF_CloseSession(self._session, status)
          if tf_session.TF_GetCode(status) != 0:
            raise RuntimeError(tf_session.TF_Message(status))
        finally:
          tf_session.TF_DeleteStatus(status)

  def __del__(self):
    self.close()
    try:
      status = tf_session.TF_NewStatus()
      if self._session is not None:
        tf_session.TF_DeleteSession(self._session, status)
        if tf_session.TF_GetCode(status) != 0:
          raise RuntimeError(tf_session.TF_Message(status))
        self._session = None
    finally:
      tf_session.TF_DeleteStatus(status)

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
    return self._graph.as_graph_def()

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
      print c.eval()
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
      print c.eval()
    # ...
    with sess.as_default():
      print c.eval()

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
  # Tensor expansions. Expects tuples of (Type, fetch_fn, feed_fn),
  # where the signatures are:
  #   fetch_fn : Type -> (list of Tensors,
  #                       lambda: list of fetched np.ndarray -> TypeVal)
  #   feed_fn  : Type, TypeVal -> list of (Tensor, value)
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
           [feed.indices, feed.values, feed.shape], feed_val))),
      # The default catches all types and performs no expansions.
      (object,
       lambda fetch: ([fetch], lambda fetched_vals: fetched_vals[0]),
       lambda feed, feed_val: [(feed, feed_val)])]
  # pylint: enable=g-long-lambda

  def run(self, fetches, feed_dict=None):
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

    Args:
      fetches: A single graph element, or a list of graph elements
        (described above).
      feed_dict: A dictionary that maps graph elements to values
        (described above).

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
    def _fetch_fn(fetch):
      for tensor_type, fetch_fn, _ in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(fetch, tensor_type):
          return fetch_fn(fetch)
      raise TypeError('Fetch argument %r has invalid type %r'
                      % (fetch, type(fetch)))

    def _feed_fn(feed, feed_val):
      for tensor_type, _, feed_fn in BaseSession._REGISTERED_EXPANSIONS:
        if isinstance(feed, tensor_type):
          return feed_fn(feed, feed_val)
      raise TypeError('Feed argument %r has invalid type %r'
                      % (feed, type(feed)))

    # Check session.
    if self._closed:
      raise RuntimeError('Attempted to use a closed Session.')

    # Validate and process fetches.
    is_list_fetch = isinstance(fetches, (list, tuple))
    if not is_list_fetch:
      fetches = [fetches]

    unique_fetch_targets = set()
    target_list = []

    fetch_info = []
    for fetch in fetches:
      subfetches, fetch_contraction_fn = _fetch_fn(fetch)
      subfetch_names = []
      for subfetch in subfetches:
        try:
          fetch_t = self.graph.as_graph_element(subfetch, allow_tensor=True,
                                                allow_operation=True)
          if isinstance(fetch_t, ops.Operation):
            target_list.append(fetch_t.name)
          else:
            subfetch_names.append(fetch_t.name)
        except TypeError as e:
          raise TypeError('Fetch argument %r of %r has invalid type %r, '
                          'must be a string or Tensor. (%s)'
                          % (subfetch, fetch, type(subfetch), e.message))
        except ValueError as e:
          raise ValueError('Fetch argument %r of %r cannot be interpreted as a '
                           'Tensor. (%s)' % (subfetch, fetch, e.message))
        except KeyError as e:
          raise ValueError('Fetch argument %r of %r cannot be interpreted as a '
                           'Tensor. (%s)' % (subfetch, fetch, e.message))
      unique_fetch_targets.update(subfetch_names)
      fetch_info.append((subfetch_names, fetch_contraction_fn))

    unique_fetch_targets = list(unique_fetch_targets)

    # Create request.
    feed_dict_string = {}

    # Validate and process feed_dict.
    if feed_dict:
      for feed, feed_val in feed_dict.iteritems():
        for subfeed, subfeed_val in _feed_fn(feed, feed_val):
          try:
            subfeed_t = self.graph.as_graph_element(subfeed, allow_tensor=True,
                                                    allow_operation=False)
          except Exception as e:
            e.message = ('Cannot interpret feed_dict key as Tensor: '
                         + e.message)
            e.args = (e.message,)
            raise e
          np_val = np.array(subfeed_val, dtype=subfeed_t.dtype.as_numpy_dtype)
          if subfeed_t.op.type == 'Placeholder':
            if not subfeed_t.get_shape().is_compatible_with(np_val.shape):
              raise ValueError(
                  'Cannot feed value of shape %r for Tensor %r, '
                  'which has shape %r'
                  % (np_val.shape, subfeed_t.name,
                     tuple(subfeed_t.get_shape().dims)))
          feed_dict_string[str(subfeed_t.name)] = np_val

    # Run request and get response.
    results = self._do_run(target_list, unique_fetch_targets, feed_dict_string)

    # User may have fetched the same tensor multiple times, but we
    # only fetch them from the runtime once.  Furthermore, they may
    # be wrapped as a tuple of tensors.  Here we map the results back
    # to what the client asked for.
    fetched_results = dict(zip(unique_fetch_targets, results))
    ret = []
    for fetch_names, fetch_contraction_fn in fetch_info:
      if fetch_names:
        fetched_vals = [fetched_results[name] for name in fetch_names]
        ret.append(fetch_contraction_fn(fetched_vals))
      else:
        ret.append(None)

    if is_list_fetch:
      return ret
    else:
      return ret[0]

  # Captures the name of a node in an error status.
  _NODEDEF_NAME_RE = re.compile(r'\[\[Node: ([^ ]*?) =')

  def _do_run(self, target_list, fetch_list, feed_dict):
    """Runs a step based on the given fetches and feeds.

    Args:
      target_list: A list of strings corresponding to names of tensors
        or operations to be run to, but not fetched.
      fetch_list: A list of strings corresponding to names of tensors to be
        fetched and operations to be run.
      feed_dict: A dictionary that maps tensor names to numpy ndarrays.

    Returns:
      A list of numpy ndarrays, corresponding to the elements of
      `fetch_list`.  If the ith element of `fetch_list` contains the
      name of an operation, the first Tensor output of that operation
      will be returned for that element.
    """
    try:
      # Ensure any changes to the graph are reflected in the runtime.
      with self._extend_lock:
        if self._graph.version > self._current_version:
          graph_def = self._graph.as_graph_def(
              from_version=self._current_version)

          try:
            status = tf_session.TF_NewStatus()
            tf_session.TF_ExtendGraph(
                self._session, graph_def.SerializeToString(), status)
            if tf_session.TF_GetCode(status) != 0:
              raise RuntimeError(tf_session.TF_Message(status))
            self._opened = True
          finally:
            tf_session.TF_DeleteStatus(status)

          self._current_version = self._graph.version

      return tf_session.TF_Run(self._session, feed_dict, fetch_list,
                               target_list)

    except tf_session.StatusNotOK as e:
      e_type, e_value, e_traceback = sys.exc_info()
      m = BaseSession._NODEDEF_NAME_RE.search(e.error_message)
      if m is not None:
        node_name = m.group(1)
        node_def = None
        try:
          op = self._graph.get_operation_by_name(node_name)
          node_def = op.node_def
        except KeyError:
          op = None
        # pylint: disable=protected-access
        raise errors._make_specific_exception(node_def, op, e.error_message,
                                              e.code)
        # pylint: enable=protected-access
      raise e_type, e_value, e_traceback


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
  print sess.run(c)
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
  (https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/config.proto)
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
      config: (Optional.) A [`ConfigProto`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/config.proto)
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
  print c.eval()
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
    print c.eval()
  ```

  @@__init__
  @@close
  """

  def __init__(self, target='', graph=None):
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
    """
    super(InteractiveSession, self).__init__(target, graph)
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
