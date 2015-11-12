"""Test that old style division works for Dimension."""
from __future__ import absolute_import
# from __future__ import division  # Intentionally skip this import
from __future__ import print_function

import tensorflow.python.platform

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class DimensionDivTest(test_util.TensorFlowTestCase):

  def testDivSucceeds(self):
    """Without from __future__ import division, __div__ should work."""
    values = [tensor_shape.Dimension(x) for x in 3, 7, 11, None]
    for x in values:
      for y in values:
        self.assertEqual((x / y).value, (x // y).value)


if __name__ == "__main__":
  googletest.main()
