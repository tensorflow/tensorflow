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

# pylint: disable=g-short-docstring-punctuation
"""## Unit tests

TensorFlow provides a convenience class inheriting from `unittest.TestCase`
which adds methods relevant to TensorFlow tests.  Here is an example:

    import tensorflow as tf


    class SquareTest(tf.test.TestCase):

      def testSquare(self):
        with self.test_session():
          x = tf.square([2, 3])
          self.assertAllEqual(x.eval(), [4, 9])


    if __name__ == '__main__':
      tf.test.main()


`tf.test.TestCase` inherits from `unittest.TestCase` but adds a few additional
methods.  We will document these methods soon.

@@main

## Utilities

@@assert_equal_graph_def
@@get_temp_dir
@@is_built_with_cuda

## Gradient checking

[`compute_gradient`](#compute_gradient) and
[`compute_gradient_error`](#compute_gradient_error) perform numerical
differentiation of graphs for comparison against registered analytic gradients.

@@compute_gradient
@@compute_gradient_error

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.util.all_util import make_all

# pylint: disable=unused-import
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import assert_equal_graph_def

from tensorflow.python.kernel_tests.gradient_checker import compute_gradient_error
from tensorflow.python.kernel_tests.gradient_checker import compute_gradient
# pylint: enable=unused-import


def main():
  """Runs all unit tests."""
  return googletest.main()


def get_temp_dir():
  """Returns a temporary directory for use during tests.

  There is no need to delete the directory after the test.

  Returns:
    The temporary directory.
  """
  return googletest.GetTempDir()


def is_built_with_cuda():
  """Returns whether TensorFlow was built with CUDA (GPU) support."""
  return test_util.IsGoogleCudaEnabled()


__all__ = make_all(__name__)
# TODO(irving,vrv): Remove once TestCase is documented
__all__.append('TestCase')
