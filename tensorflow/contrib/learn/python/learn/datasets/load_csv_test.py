# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn import datasets
from tensorflow.python.platform import test


class LoadCsvTest(test.TestCase):
  """Test load csv functions."""

  def testIris(self):
    iris = datasets.load_iris()
    self.assertTupleEqual(iris.data.shape, (150, 4))
    self.assertTupleEqual(iris.target.shape, (150,))

  def testBoston(self):
    boston = datasets.load_boston()
    self.assertTupleEqual(boston.data.shape, (506, 13))
    self.assertTupleEqual(boston.target.shape, (506,))


if __name__ == "__main__":
  test.main()
