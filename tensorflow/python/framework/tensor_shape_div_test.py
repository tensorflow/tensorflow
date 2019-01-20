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

"""Test that old style division works for Dimension."""
from __future__ import absolute_import
# from __future__ import division  # Intentionally skip this import
from __future__ import print_function

import six

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class DimensionDivTest(test_util.TensorFlowTestCase):

  def testDivSucceeds(self):
    """Without from __future__ import division, __div__ should work."""
    if six.PY2:  # Old division exists only in Python 2
      values = [tensor_shape.Dimension(x) for x in (3, 7, 11, None)]
      for x in values:
        for y in values:
          self.assertEqual((x / y).value, (x // y).value)


if __name__ == "__main__":
  googletest.main()
