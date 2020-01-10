"""Tests for tensorflow.python.framework.errors."""
import tensorflow.python.platform

import warnings

import tensorflow as tf

from tensorflow.core.lib.core import error_codes_pb2

class ErrorsTest(tf.test.TestCase):

  def testUniqueClassForEachErrorCode(self):
    for error_code, exc_type in [
        (tf.errors.CANCELLED, tf.errors.CancelledError),
        (tf.errors.UNKNOWN, tf.errors.UnknownError),
        (tf.errors.INVALID_ARGUMENT, tf.errors.InvalidArgumentError),
        (tf.errors.DEADLINE_EXCEEDED, tf.errors.DeadlineExceededError),
        (tf.errors.NOT_FOUND, tf.errors.NotFoundError),
        (tf.errors.ALREADY_EXISTS, tf.errors.AlreadyExistsError),
        (tf.errors.PERMISSION_DENIED, tf.errors.PermissionDeniedError),
        (tf.errors.UNAUTHENTICATED, tf.errors.UnauthenticatedError),
        (tf.errors.RESOURCE_EXHAUSTED, tf.errors.ResourceExhaustedError),
        (tf.errors.FAILED_PRECONDITION, tf.errors.FailedPreconditionError),
        (tf.errors.ABORTED, tf.errors.AbortedError),
        (tf.errors.OUT_OF_RANGE, tf.errors.OutOfRangeError),
        (tf.errors.UNIMPLEMENTED, tf.errors.UnimplementedError),
        (tf.errors.INTERNAL, tf.errors.InternalError),
        (tf.errors.UNAVAILABLE, tf.errors.UnavailableError),
        (tf.errors.DATA_LOSS, tf.errors.DataLossError),
        ]:
      # pylint: disable=protected-access
      self.assertTrue(isinstance(
          tf.errors._make_specific_exception(None, None, None, error_code),
          exc_type))
      # pylint: enable=protected-access

  def testKnownErrorClassForEachErrorCodeInProto(self):
    for error_code in error_codes_pb2.Code.values():
      # pylint: disable=line-too-long
      if error_code in (error_codes_pb2.OK,
                        error_codes_pb2.DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_):
        continue
      # pylint: enable=line-too-long
      with warnings.catch_warnings(record=True) as w:
        # pylint: disable=protected-access
        exc = tf.errors._make_specific_exception(None, None, None, error_code)
        # pylint: enable=protected-access
      self.assertEqual(0, len(w))  # No warning is raised.
      self.assertTrue(isinstance(exc, tf.errors.OpError))
      self.assertTrue(tf.errors.OpError in exc.__class__.__bases__)

  def testUnknownErrorCodeCausesWarning(self):
    with warnings.catch_warnings(record=True) as w:
      # pylint: disable=protected-access
      exc = tf.errors._make_specific_exception(None, None, None, 37)
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown error code: 37" in str(w[0].message))
    self.assertTrue(isinstance(exc, tf.errors.OpError))


if __name__ == "__main__":
  tf.test.main()
