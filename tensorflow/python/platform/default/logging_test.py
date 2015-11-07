from tensorflow.python.platform.default import _googletest as googletest
from tensorflow.python.platform.default import _logging as logging


class EventLoaderTest(googletest.TestCase):

  def test_log(self):
    # Just check that logging works without raising an exception.
    logging.error("test log message")


if __name__ == "__main__":
  googletest.main()
