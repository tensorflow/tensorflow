"""Tests for SWIG wrapped brain::Status."""

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import googletest


class StatusTest(googletest.TestCase):

  def testDefaultOk(self):
    status = pywrap_tensorflow.Status()
    self.assertTrue(status.ok())

  def testCodeAndMessage(self):
    status = pywrap_tensorflow.Status(error_codes_pb2.INVALID_ARGUMENT, 'foo')
    self.assertEqual(error_codes_pb2.INVALID_ARGUMENT, status.code())
    self.assertEqual('foo', status.error_message())

  def testToString(self):
    status = pywrap_tensorflow.Status()
    # .ToString was remapped in the .swig file, hence will not work
    # self.assertIn('OK', status.ToString())
    self.assertIn('OK', str(status))

  def testException(self):
    with self.assertRaises(pywrap_tensorflow.StatusNotOK) as context:
      pywrap_tensorflow.NotOkay()
    self.assertEqual(context.exception.code, error_codes_pb2.INVALID_ARGUMENT)
    self.assertEqual(context.exception.error_message, 'Testing 1 2 3')
    self.assertEqual(None, pywrap_tensorflow.Okay(),
                     'Status wrapper should not return anything upon OK.')


if __name__ == '__main__':
  googletest.main()
