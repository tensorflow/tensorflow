# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=invalid-name
"""Test utils for tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import math
import random
import re
import tempfile
import threading

import numpy as np
import six

_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:
  _portpicker_import_error = _error
  portpicker = None

# pylint: disable=g-import-not-at-top
from google.protobuf import descriptor_pool
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import device_lib
from tensorflow.python.client import session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
from tensorflow.python.util.protobuf import compare


def gpu_device_name():
  """Returns the name of a GPU device if available or the empty string."""
  for x in device_lib.list_local_devices():
    if x.device_type == "GPU" or x.device_type == "SYCL":
      return x.name
  return ""


def assert_ops_in_graph(expected_ops, graph):
  """Assert all expected operations are found.

  Args:
    expected_ops: `dict<string, string>` of op name to op type.
    graph: Graph to check.
  Returns:
    `dict<string, node>` of node name to node.

  Raises:
    ValueError: If the expected ops are not present in the graph.
  """
  actual_ops = {}
  gd = graph.as_graph_def()
  for node in gd.node:
    if node.name in expected_ops:
      if expected_ops[node.name] != node.op:
        raise ValueError("Expected op for node %s is different. %s vs %s" %
                         (node.name, expected_ops[node.name], node.op))
      actual_ops[node.name] = node
  if set(expected_ops.keys()) != set(actual_ops.keys()):
    raise ValueError("Not all expected ops are present. Expected %s, found %s" %
                     (expected_ops.keys(), actual_ops.keys()))
  return actual_ops


def assert_equal_graph_def(actual, expected, checkpoint_v2=False):
  """Asserts that two `GraphDef`s are (mostly) the same.

  Compares two `GraphDef` protos for equality, ignoring versions and ordering of
  nodes, attrs, and control inputs.  Node names are used to match up nodes
  between the graphs, so the naming of nodes must be consistent.

  Args:
    actual: The `GraphDef` we have.
    expected: The `GraphDef` we expected.
    checkpoint_v2: boolean determining whether to ignore randomized attribute
        values that appear in V2 checkpoints.

  Raises:
    AssertionError: If the `GraphDef`s do not match.
    TypeError: If either argument is not a `GraphDef`.
  """
  if not isinstance(actual, graph_pb2.GraphDef):
    raise TypeError("Expected tf.GraphDef for actual, got %s" %
                    type(actual).__name__)
  if not isinstance(expected, graph_pb2.GraphDef):
    raise TypeError("Expected tf.GraphDef for expected, got %s" %
                    type(expected).__name__)

  if checkpoint_v2:
    _strip_checkpoint_v2_randomized(actual)
    _strip_checkpoint_v2_randomized(expected)

  diff = pywrap_tensorflow.EqualGraphDefWrapper(actual.SerializeToString(),
                                                expected.SerializeToString())
  if diff:
    raise AssertionError(compat.as_str(diff))


def assert_meta_graph_protos_equal(tester, a, b):
  """Compares MetaGraphDefs `a` and `b` in unit test class `tester`."""
  # Carefully check the collection_defs
  tester.assertEqual(set(a.collection_def), set(b.collection_def))
  collection_keys = a.collection_def.keys()
  for k in collection_keys:
    a_value = a.collection_def[k]
    b_value = b.collection_def[k]
    proto_type = ops.get_collection_proto_type(k)
    if proto_type:
      a_proto = proto_type()
      b_proto = proto_type()
      # Number of entries in the collections is the same
      tester.assertEqual(len(a_value.bytes_list.value),
                         len(b_value.bytes_list.value))
      for (a_value_item, b_value_item) in zip(
          a_value.bytes_list.value,
          b_value.bytes_list.value):
        a_proto.ParseFromString(a_value_item)
        b_proto.ParseFromString(b_value_item)
        tester.assertProtoEquals(a_proto, b_proto)
    else:
      tester.assertEquals(a_value, b_value)
  # Compared the fields directly, remove their raw values from the
  # proto comparison below.
  a.ClearField("collection_def")
  b.ClearField("collection_def")
  tester.assertProtoEquals(a, b)


# Matches attributes named via _SHARDED_SUFFIX in
# tensorflow/python/training/saver.py
_SHARDED_SAVE_OP_PATTERN = "_temp_[0-9a-z]{32}/part"


def _strip_checkpoint_v2_randomized(graph_def):
  for node in graph_def.node:
    delete_keys = []
    for attr_key in node.attr:
      attr_tensor_value = node.attr[attr_key].tensor
      if attr_tensor_value and len(attr_tensor_value.string_val) == 1:
        attr_tensor_string_value = attr_tensor_value.string_val[0]
        if (attr_tensor_string_value and
            re.match(_SHARDED_SAVE_OP_PATTERN, attr_tensor_string_value)):
          delete_keys.append(attr_key)
    for attr_key in delete_keys:
      del node.attr[attr_key]


def IsGoogleCudaEnabled():
  return pywrap_tensorflow.IsGoogleCudaEnabled()


def CudaSupportsHalfMatMulAndConv():
  return pywrap_tensorflow.CudaSupportsHalfMatMulAndConv()


def NHWCToNCHW(input_tensor):
  """Converts the input from the NHWC format to NCHW.

  Args:
    input_tensor: a 4- or 5-D tensor, or an array representing shape

  Returns:
    converted tensor or shape array
  """
  # tensor dim -> new axis order
  new_axes = {
      4: [0, 3, 1, 2],
      5: [0, 4, 1, 2, 3]
  }
  if isinstance(input_tensor, ops.Tensor):
    ndims = input_tensor.shape.ndims
    return array_ops.transpose(input_tensor, new_axes[ndims])
  else:
    ndims = len(input_tensor)
    return [input_tensor[a] for a in new_axes[ndims]]


def NCHWToNHWC(input_tensor):
  """Converts the input from the NCHW format to NHWC.

  Args:
    input_tensor: a 4- or 5-D tensor, or an array representing shape

  Returns:
    converted tensor or shape array
  """
  # tensor dim -> new axis order
  new_axes = {
      4: [0, 2, 3, 1],
      5: [0, 2, 3, 4, 1]
  }
  if isinstance(input_tensor, ops.Tensor):
    ndims = input_tensor.shape.ndims
    return array_ops.transpose(input_tensor, new_axes[ndims])
  else:
    ndims = len(input_tensor)
    return [input_tensor[a] for a in new_axes[ndims]]


class TensorFlowTestCase(googletest.TestCase):
  """Base class for tests that need to test TensorFlow.
  """

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(TensorFlowTestCase, self).__init__(methodName)
    self._threads = []
    self._tempdir = None
    self._cached_session = None

  def setUp(self):
    self._ClearCachedSession()
    random.seed(random_seed.DEFAULT_GRAPH_SEED)
    np.random.seed(random_seed.DEFAULT_GRAPH_SEED)
    ops.reset_default_graph()
    ops.get_default_graph().seed = random_seed.DEFAULT_GRAPH_SEED

  def tearDown(self):
    for thread in self._threads:
      self.assertFalse(thread.is_alive(), "A checkedThread did not terminate")

    self._ClearCachedSession()

  def _ClearCachedSession(self):
    if self._cached_session is not None:
      self._cached_session.close()
      self._cached_session = None

  def get_temp_dir(self):
    """Returns a unique temporary directory for the test to use.

    If you call this method multiple times during in a test, it will return the
    same folder. However, across different runs the directories will be
    different. This will ensure that across different runs tests will not be
    able to pollute each others environment.
    If you need multiple unique directories within a single test, you should
    use tempfile.mkdtemp as follows:
      tempfile.mkdtemp(dir=self.get_temp_dir()):

    Returns:
      string, the path to the unique temporary directory created for this test.
    """
    if not self._tempdir:
      self._tempdir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

  def _AssertProtoEquals(self, a, b):
    """Asserts that a and b are the same proto.

    Uses ProtoEq() first, as it returns correct results
    for floating point attributes, and then use assertProtoEqual()
    in case of failure as it provides good error messages.

    Args:
      a: a proto.
      b: another proto.
    """
    if not compare.ProtoEq(a, b):
      compare.assertProtoEqual(self, a, b, normalize_numbers=True)

  def assertProtoEquals(self, expected_message_maybe_ascii, message):
    """Asserts that message is same as parsed expected_message_ascii.

    Creates another prototype of message, reads the ascii message into it and
    then compares them using self._AssertProtoEqual().

    Args:
      expected_message_maybe_ascii: proto message in original or ascii form.
      message: the message to validate.
    """

    if isinstance(expected_message_maybe_ascii, type(message)):
      expected_message = expected_message_maybe_ascii
      self._AssertProtoEquals(expected_message, message)
    elif isinstance(expected_message_maybe_ascii, str):
      expected_message = type(message)()
      text_format.Merge(expected_message_maybe_ascii, expected_message,
                        descriptor_pool=descriptor_pool.Default())
      self._AssertProtoEquals(expected_message, message)
    else:
      assert False, ("Can't compare protos of type %s and %s" %
                     (type(expected_message_maybe_ascii), type(message)))

  def assertProtoEqualsVersion(
      self,
      expected,
      actual,
      producer=versions.GRAPH_DEF_VERSION,
      min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER):
    expected = "versions { producer: %d min_consumer: %d };\n%s" % (
        producer, min_consumer, expected)
    self.assertProtoEquals(expected, actual)

  def assertStartsWith(self, actual, expected_start, msg=None):
    """Assert that actual.startswith(expected_start) is True.

    Args:
      actual: str
      expected_start: str
      msg: Optional message to report on failure.
    """
    if not actual.startswith(expected_start):
      fail_msg = "%r does not start with %r" % (actual, expected_start)
      fail_msg += " : %r" % (msg) if msg else ""
      self.fail(fail_msg)

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def test_session(self,
                   graph=None,
                   config=None,
                   use_gpu=False,
                   force_gpu=False):
    """Returns a TensorFlow Session for use in executing tests.

    This method should be used for all functional tests.

    This method behaves different than session.Session: for performance reasons
    `test_session` will by default (if `graph` is None) reuse the same session
    across tests. This means you may want to either call the function
    `reset_default_graph()` before tests, or if creating an explicit new graph,
    pass it here (simply setting it with `as_default()` won't do it), which will
    trigger the creation of a new session.

    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/gpu:0`. Otherwise, if `use_gpu`
    is True, TensorFlow tries to run as many ops on the GPU as possible. If both
    `force_gpu and `use_gpu` are False, all ops are pinned to the CPU.

    Example:

      class MyOperatorTest(test_util.TensorFlowTestCase):
        def testMyOperator(self):
          with self.test_session(use_gpu=True):
            valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = MyOperator(valid_input).eval()
            self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
            invalid_input = [-1.0, 2.0, 7.0]
            with self.assertRaisesOpError("negative input not supported"):
              MyOperator(invalid_input).eval()

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/gpu:0`.

    Returns:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if self.id().endswith(".test_session"):
      self.skipTest("Not a test.")

    def prepare_config(config):
      """Returns a config for sessions.

      Args:
        config: An optional config_pb2.ConfigProto to use to configure the
          session.
      Returns:
        A config_pb2.ConfigProto object.
      """
      if config is None:
        config = config_pb2.ConfigProto()
        config.allow_soft_placement = not force_gpu
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
      elif force_gpu and config.allow_soft_placement:
        config = config_pb2.ConfigProto().CopyFrom(config)
        config.allow_soft_placement = False
      # Don't perform optimizations for tests so we don't inadvertently run
      # gpu ops on cpu
      config.graph_options.optimizer_options.opt_level = -1
      return config

    if graph is None:
      if self._cached_session is None:
        self._cached_session = session.Session(
            graph=None, config=prepare_config(config))
      sess = self._cached_session
      with sess.graph.as_default(), sess.as_default():
        if force_gpu:
          # Use the name of an actual device if one is detected, or '/gpu:0'
          # otherwise
          gpu_name = gpu_device_name()
          if not gpu_name:
            gpu_name = "/gpu:0"
          with sess.graph.device(gpu_name):
            yield sess
        elif use_gpu:
          yield sess
        else:
          with sess.graph.device("/cpu:0"):
            yield sess
    else:
      with session.Session(graph=graph, config=prepare_config(config)) as sess:
        if force_gpu:
          # Use the name of an actual device if one is detected, or '/gpu:0'
          # otherwise
          gpu_name = gpu_device_name()
          if not gpu_name:
            gpu_name = "/gpu:0"
          with sess.graph.device(gpu_name):
            yield sess
        elif use_gpu:
          yield sess
        else:
          with sess.graph.device("/cpu:0"):
            yield sess

  # pylint: enable=g-doc-return-or-yield

  class _CheckedThread(object):
    """A wrapper class for Thread that asserts successful completion.

    This class should be created using the TensorFlowTestCase.checkedThread()
    method.
    """

    def __init__(self, testcase, target, args=None, kwargs=None):
      """Constructs a new instance of _CheckedThread.

      Args:
        testcase: The TensorFlowTestCase for which this thread is being created.
        target: A callable object representing the code to be executed in the
          thread.
        args: A tuple of positional arguments that will be passed to target.
        kwargs: A dictionary of keyword arguments that will be passed to target.
      """
      self._testcase = testcase
      self._target = target
      self._args = () if args is None else args
      self._kwargs = {} if kwargs is None else kwargs
      self._thread = threading.Thread(target=self._protected_run)
      self._exception = None

    def _protected_run(self):
      """Target for the wrapper thread. Sets self._exception on failure."""
      try:
        self._target(*self._args, **self._kwargs)
      except Exception as e:  # pylint: disable=broad-except
        self._exception = e

    def start(self):
      """Starts the thread's activity.

      This must be called at most once per _CheckedThread object. It arranges
      for the object's target to be invoked in a separate thread of control.
      """
      self._thread.start()

    def join(self):
      """Blocks until the thread terminates.

      Raises:
        self._testcase.failureException: If the thread terminates with due to
          an exception.
      """
      self._thread.join()
      if self._exception is not None:
        self._testcase.fail("Error in checkedThread: %s" % str(self._exception))

    def is_alive(self):
      """Returns whether the thread is alive.

      This method returns True just before the run() method starts
      until just after the run() method terminates.

      Returns:
        True if the thread is alive, otherwise False.
      """
      return self._thread.is_alive()

  def checkedThread(self, target, args=None, kwargs=None):
    """Returns a Thread wrapper that asserts 'target' completes successfully.

    This method should be used to create all threads in test cases, as
    otherwise there is a risk that a thread will silently fail, and/or
    assertions made in the thread will not be respected.

    Args:
      target: A callable object to be executed in the thread.
      args: The argument tuple for the target invocation. Defaults to ().
      kwargs: A dictionary of keyword arguments for the target invocation.
        Defaults to {}.

    Returns:
      A wrapper for threading.Thread that supports start() and join() methods.
    """
    ret = TensorFlowTestCase._CheckedThread(self, target, args, kwargs)
    self._threads.append(ret)
    return ret

# pylint: enable=invalid-name

  def assertNear(self, f1, f2, err, msg=None):
    """Asserts that two floats are near each other.

    Checks that |f1 - f2| < err and asserts a test failure
    if not.

    Args:
      f1: A float value.
      f2: A float value.
      err: A float value.
      msg: An optional string message to append to the failure message.
    """
    self.assertTrue(
        math.fabs(f1 - f2) <= err,
        "%f != %f +/- %f%s" % (f1, f2, err, " (%s)" % msg
                               if msg is not None else ""))

  def assertArrayNear(self, farray1, farray2, err):
    """Asserts that two float arrays are near each other.

    Checks that for all elements of farray1 and farray2
    |f1 - f2| < err.  Asserts a test failure if not.

    Args:
      farray1: a list of float values.
      farray2: a list of float values.
      err: a float value.
    """
    self.assertEqual(len(farray1), len(farray2))
    for f1, f2 in zip(farray1, farray2):
      self.assertNear(float(f1), float(f2), err)

  def _NDArrayNear(self, ndarray1, ndarray2, err):
    return np.linalg.norm(ndarray1 - ndarray2) < err

  def assertNDArrayNear(self, ndarray1, ndarray2, err):
    """Asserts that two numpy arrays have near values.

    Args:
      ndarray1: a numpy ndarray.
      ndarray2: a numpy ndarray.
      err: a float. The maximum absolute difference allowed.
    """
    self.assertTrue(self._NDArrayNear(ndarray1, ndarray2, err))

  def _GetNdArray(self, a):
    if not isinstance(a, np.ndarray):
      a = np.array(a)
    return a

  def _assertArrayLikeAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    self.assertEqual(a.shape, b.shape, "Shape mismatch: expected %s, got %s." %
                     (a.shape, b.shape))
    if not np.allclose(a, b, rtol=rtol, atol=atol):
      # Prints more details than np.testing.assert_allclose.
      #
      # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
      # checks whether two arrays are element-wise equal within a
      # tolerance. The relative difference (rtol * abs(b)) and the
      # absolute difference atol are added together to compare against
      # the absolute difference between a and b.  Here, we want to
      # print out which elements violate such conditions.
      cond = np.logical_or(
          np.abs(a - b) > atol + rtol * np.abs(b), np.isnan(a) != np.isnan(b))
      if a.ndim:
        x = a[np.where(cond)]
        y = b[np.where(cond)]
        print("not close where = ", np.where(cond))
      else:
        # np.where is broken for scalars
        x, y = a, b
      print("not close lhs = ", x)
      print("not close rhs = ", y)
      print("not close dif = ", np.abs(x - y))
      print("not close tol = ", atol + rtol * np.abs(y))
      print("dtype = %s, shape = %s" % (a.dtype, a.shape))
      np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)

  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6):
    """Asserts that two numpy arrays, or dicts of same, have near values.

    This does not support nested dicts.

    Args:
      a: A numpy ndarray (or anything can be converted to one), or dict of same.
        Must be a dict iff `b` is a dict.
      b: A numpy ndarray (or anything can be converted to one), or dict of same.
        Must be a dict iff `a` is a dict.
      rtol: relative tolerance.
      atol: absolute tolerance.

    Raises:
      ValueError: if only one of `a` and `b` is a dict.
    """
    is_a_dict = isinstance(a, dict)
    if is_a_dict != isinstance(b, dict):
      raise ValueError("Can't compare dict to non-dict, %s vs %s." % (a, b))
    if is_a_dict:
      self.assertItemsEqual(
          a.keys(), b.keys(),
          msg="mismatched keys, expected %s, got %s" % (a.keys(), b.keys()))
      for k in a:
        self._assertArrayLikeAllClose(
            a[k], b[k], rtol=rtol, atol=atol,
            msg="%s: expected %s, got %s." % (k, a, b))
    else:
      self._assertArrayLikeAllClose(a, b, rtol=rtol, atol=atol)

  def assertAllCloseAccordingToType(self,
                                    a,
                                    b,
                                    rtol=1e-6,
                                    atol=1e-6,
                                    float_rtol=1e-6,
                                    float_atol=1e-6,
                                    half_rtol=1e-3,
                                    half_atol=1e-3):
    """Like assertAllClose, but also suitable for comparing fp16 arrays.

    In particular, the tolerance is reduced to 1e-3 if at least
    one of the arguments is of type float16.

    Args:
      a: a numpy ndarray or anything can be converted to one.
      b: a numpy ndarray or anything can be converted to one.
      rtol: relative tolerance.
      atol: absolute tolerance.
      float_rtol: relative tolerance for float32.
      float_atol: absolute tolerance for float32.
      half_rtol: relative tolerance for float16.
      half_atol: absolute tolerance for float16.
    """
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    if (a.dtype == np.float32 or b.dtype == np.float32 or
        a.dtype == np.complex64 or b.dtype == np.complex64):
      rtol = max(rtol, float_rtol)
      atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
      rtol = max(rtol, half_rtol)
      atol = max(atol, half_atol)

    self.assertAllClose(a, b, rtol=rtol, atol=atol)

  def assertAllEqual(self, a, b):
    """Asserts that two numpy arrays have the same values.

    Args:
      a: a numpy ndarray or anything can be converted to one.
      b: a numpy ndarray or anything can be converted to one.
    """
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    self.assertEqual(a.shape, b.shape, "Shape mismatch: expected %s, got %s." %
                     (a.shape, b.shape))
    same = (a == b)

    if a.dtype == np.float32 or a.dtype == np.float64:
      same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    if not np.all(same):
      # Prints more details than np.testing.assert_array_equal.
      diff = np.logical_not(same)
      if a.ndim:
        x = a[np.where(diff)]
        y = b[np.where(diff)]
        print("not equal where = ", np.where(diff))
      else:
        # np.where is broken for scalars
        x, y = a, b
      print("not equal lhs = ", x)
      print("not equal rhs = ", y)
      np.testing.assert_array_equal(a, b)

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def assertRaisesWithPredicateMatch(self, exception_type,
                                     expected_err_re_or_predicate):
    """Returns a context manager to enclose code expected to raise an exception.

    If the exception is an OpError, the op stack is also included in the message
    predicate search.

    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in exception and
        returns True (success) or False (please fail the test). Otherwise, the
        error message is expected to match this regular expression partially.

    Returns:
      A context manager to surround code that is expected to raise an
      exception.
    """
    if callable(expected_err_re_or_predicate):
      predicate = expected_err_re_or_predicate
    else:

      def predicate(e):
        err_str = e.message if isinstance(e, errors.OpError) else str(e)
        op = e.op if isinstance(e, errors.OpError) else None
        while op is not None:
          err_str += "\nCaused by: " + op.name
          op = op._original_op  # pylint: disable=protected-access
        logging.info("Searching within error strings: '%s' within '%s'",
                     expected_err_re_or_predicate, err_str)
        return re.search(expected_err_re_or_predicate, err_str)

    try:
      yield
      self.fail(exception_type.__name__ + " not raised")
    except Exception as e:  # pylint: disable=broad-except
      if not isinstance(e, exception_type) or not predicate(e):
        raise AssertionError("Exception of type %s: %s" % (str(type(e)),
                                                           str(e)))

  # pylint: enable=g-doc-return-or-yield

  def assertRaisesOpError(self, expected_err_re_or_predicate):
    return self.assertRaisesWithPredicateMatch(errors.OpError,
                                               expected_err_re_or_predicate)

  def assertShapeEqual(self, np_array, tf_tensor):
    """Asserts that a Numpy ndarray and a TensorFlow tensor have the same shape.

    Args:
      np_array: A Numpy ndarray or Numpy scalar.
      tf_tensor: A Tensor.

    Raises:
      TypeError: If the arguments have the wrong type.
    """
    if not isinstance(np_array, (np.ndarray, np.generic)):
      raise TypeError("np_array must be a Numpy ndarray or Numpy scalar")
    if not isinstance(tf_tensor, ops.Tensor):
      raise TypeError("tf_tensor must be a Tensor")
    self.assertAllEqual(np_array.shape, tf_tensor.get_shape().as_list())

  def assertDeviceEqual(self, device1, device2):
    """Asserts that the two given devices are the same.

    Args:
      device1: A string device name or TensorFlow `DeviceSpec` object.
      device2: A string device name or TensorFlow `DeviceSpec` object.
    """
    device1 = pydev.canonical_name(device1)
    device2 = pydev.canonical_name(device2)
    self.assertEqual(device1, device2,
                     "Devices %s and %s are not equal" % (device1, device2))

  # Fix Python 3 compatibility issues
  if six.PY3:
    # pylint: disable=invalid-name

    # Silence a deprecation warning
    assertRaisesRegexp = googletest.TestCase.assertRaisesRegex

    # assertItemsEqual is assertCountEqual as of 3.2.
    assertItemsEqual = googletest.TestCase.assertCountEqual

    # pylint: enable=invalid-name


def create_local_cluster(num_workers, num_ps, protocol="grpc"):
  """Create and start local servers and return the associated `Server` objects.

  Example:
  ```python
  workers, _ = tf.test.create_local_cluster(num_workers=2, num_ps=2)

  worker_sessions = [tf.Session(w.target) for w in workers]

  with tf.device("/job:ps/task:0"):
    ...
  with tf.device("/job:ps/task:1"):
    ...
  with tf.device("/job:worker/task:0"):
    ...
  with tf.device("/job:worker/task:1"):
    ...

  worker_sessions[0].run(...)
  ```

  Args:
    num_workers: Number of worker servers to start.
    num_ps: Number of PS servers to start.
    protocol: Communication protocol.  Allowed values are documented in
      the documentation of `tf.train.Server`.

  Returns:
    A tuple `(worker_servers, ps_servers)`.  `worker_servers` is a list
    of `num_workers` objects of type `tf.train.Server` (all running locally);
    and `ps_servers` is a list of `num_ps` objects of similar type.

  Raises:
    ImportError: if portpicker module was not found at load time
  """
  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs, job_name="worker", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs, job_name="ps", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_ps)
  ]

  return workers, ps_servers
