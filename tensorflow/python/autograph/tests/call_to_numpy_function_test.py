# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Simple call to a whitelisted Numpy function.

The call should be wrapped in py_func.
"""

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.autograph.tests import reference_test_base


def f():
  np.random.seed(1)
  return 2 * np.random.binomial(1, 0.5, size=(10,)) - 1


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertFunctionMatchesEager(f)


if __name__ == '__main__':
  tf.test.main()
