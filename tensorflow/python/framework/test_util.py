# pylint: disable=invalid-name
"""Test utils for tensorflow."""
import contextlib
import math
import re
import threading

import tensorflow.python.platform

import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import graph_util
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import logging
from tensorflow.python.util.protobuf import compare


def IsGoogleCudaEnabled():
  return pywrap_tensorflow.IsGoogleCudaEnabled()


class TensorFlowTestCase(googletest.TestCase):
  """Root class for tests that need to test tensor flow.
  """

  def __init__(self, methodName="runTest"):
    super(TensorFlowTestCase, self).__init__(methodName)
    self._threads = []
    self._tempdir = None
    self._cached_session = None

  def setUp(self):
    self._ClearCachedSession()
    ops.reset_default_graph()

  def tearDown(self):
    for thread in self._threads:
      self.assertFalse(thread.is_alive(), "A checkedThread did not terminate")
    self._ClearCachedSession()

  def _ClearCachedSession(self):
    if self._cached_session is not None:
      self._cached_session.close()
      self._cached_session = None

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = googletest.GetTempDir()
    return self._tempdir

  def _AssertProtoEquals(self, a, b):
    """Asserts that a and b are the same proto.

    Uses Proto2Cmp() first, as it returns correct results
    for floating point attributes, and then use assertProto2Equal()
    in case of failure as it provides good error messages.

    Args:
      a: a proto.
      b: another proto.
    """
    if compare.Proto2Cmp(a, b) != 0:
      compare.assertProto2Equal(self, a, b, normalize_numbers=True)

  def assertProtoEquals(self, expected_message_maybe_ascii, message):
    """Asserts that message is same as parsed expected_message_ascii.

    Creates another prototype of message, reads the ascii message into it and
    then compares them using self._AssertProtoEqual().

    Args:
      expected_message_maybe_ascii: proto message in original or ascii form
      message: the message to validate
    """

    if type(expected_message_maybe_ascii) == type(message):
      expected_message = expected_message_maybe_ascii
      self._AssertProtoEquals(expected_message, message)
    elif isinstance(expected_message_maybe_ascii, str):
      expected_message = type(message)()
      text_format.Merge(expected_message_maybe_ascii, expected_message)
      self._AssertProtoEquals(expected_message, message)
    else:
      assert False, ("Can't compare protos of type " +
                     type(expected_message_maybe_ascii) + " and " +
                     type(message))

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
    def prepare_config(config):
      if config is None:
        config = config_pb2.ConfigProto()
        config.allow_soft_placement = not force_gpu
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
      elif force_gpu and config.allow_soft_placement:
        config = config_pb2.ConfigProto().CopyFrom(config)
        config.allow_soft_placement = False
      return config

    if graph is None:
      if self._cached_session is None:
        self._cached_session = session.Session(graph=None,
                                               config=prepare_config(config))
      sess = self._cached_session
      with sess.graph.as_default(), sess.as_default():
        if force_gpu:
          with sess.graph.device("/gpu:0"):
            yield sess
        elif use_gpu:
          yield sess
        else:
          with sess.graph.device(graph_util.pin_to_cpu):
            yield sess
    else:
      with session.Session(graph=graph, config=prepare_config(config)) as sess:
        if force_gpu:
          with sess.graph.device("/gpu:0"):
            yield sess
        elif use_gpu:
          yield sess
        else:
          with sess.graph.device(graph_util.pin_to_cpu):
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
# pylint: disable=broad-except
      except Exception as e:
        # pylint: enable=broad-except
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
        self._testcase.fail(
            "Error in checkedThread: %s" % str(self._exception))

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

  def assertNear(self, f1, f2, err):
    """Asserts that two floats are near each other.

    Checks that |f1 - f2| < err and asserts a test failure
    if not.

    Args:
      f1: a float value.
      f2: a float value.
      err: a float value.
    """
    self.assertTrue(math.fabs(f1 - f2) < err)

  def assertArrayNear(self, farray1, farray2, err):
    """Asserts that two float arrays are near each other.

    Checks that for all elements of farray1 and farray2
    |f1 - f2| < err.  Asserts a test failure if not.

    Args:
      farray1: a list of float values.
      farray2: a list of float values.
      err: a float value.
    """
    for f1, f2 in zip(farray1, farray2):
      self.assertNear(f1, f2, err)

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

  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6):
    """Asserts that two numpy arrays have near values.

    Args:
      a: a numpy ndarray or anything can be converted to one.
      b: a numpy ndarray or anything can be converted to one.
      rtol: relative tolerance
      atol: absolute tolerance
    """
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    self.assertEqual(
        a.shape, b.shape,
        "Shape mismatch: expected %s, got %s." % (a.shape, b.shape))
    if not np.allclose(a, b, rtol=rtol, atol=atol):
      # Prints more details than np.testing.assert_allclose.
      #
      # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
      # checks whether two arrays are element-wise equal within a
      # tolerance. The relative difference (rtol * abs(b)) and the
      # absolute difference atol are added together to compare against
      # the absolute difference between a and b.  Here, we want to
      # print out which elements violate such conditions.
      cond = np.abs(a - b) > atol + rtol * np.abs(b)
      if a.ndim:
        x = a[np.where(cond)]
        y = b[np.where(cond)]
        print "not close where = ", np.where(cond)
      else:
        # np.where is broken for scalars
        x, y = a, b
      print "not close lhs = ", x
      print "not close rhs = ", y
      print "not close dif = ", np.abs(x - y)
      print "not close tol = ", atol + rtol * np.abs(y)
      np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

  def assertAllEqual(self, a, b):
    """Asserts that two numpy arrays have the same values.

    Args:
      a: a numpy ndarray or anything can be converted to one.
      b: a numpy ndarray or anything can be converted to one.
    """
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    self.assertEqual(
        a.shape, b.shape,
        "Shape mismatch: expected %s, got %s." % (a.shape, b.shape))
    same = (a == b)

    if a.dtype == np.float32 or a.dtype == np.float64:
      same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    if not np.all(same):
      # Prints more details than np.testing.assert_array_equal.
      diff = np.logical_not(same)
      if a.ndim:
        x = a[np.where(diff)]
        y = b[np.where(diff)]
        print "not equal where = ", np.where(diff)
      else:
        # np.where is broken for scalars
        x, y = a, b
      print "not equal lhs = ", x
      print "not equal rhs = ", y
      np.testing.assert_array_equal(a, b)

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def assertRaisesWithPredicateMatch(self, exception_type,
                                     expected_err_re_or_predicate):
    """Returns a context manager to enclose code expected to raise an exception.

    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in OpError exception and
        returns True (success) or False (please fail the test). Otherwise, the
        error message is expected to match this regular expression partially.

    Returns:
      A context manager to surround code that is expected to raise an
      errors.OpError exception.
    """
    if callable(expected_err_re_or_predicate):
      predicate = expected_err_re_or_predicate
    else:
      def predicate(e):
        err_str = e.message
        op = e.op
        while op is not None:
          err_str += "\nCaused by: " + op.name
          op = op._original_op
        logging.info("Searching within error strings: '%s' within '%s'",
                     expected_err_re_or_predicate, err_str)
        return re.search(expected_err_re_or_predicate, err_str)
    try:
      yield
      self.fail(exception_type.__name__ + " not raised")
# pylint: disable=broad-except
    except Exception as e:
      # pylint: enable=broad-except
      if not isinstance(e, exception_type) or not predicate(e):
        raise AssertionError(e)
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
