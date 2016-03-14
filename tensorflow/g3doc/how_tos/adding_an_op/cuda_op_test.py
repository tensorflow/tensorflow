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
"""Test for version 1 of the zero_out op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.g3doc.how_tos.adding_an_op import cuda_op


class AddOneTest(tf.test.TestCase):

  def test(self):
    if tf.test.is_built_with_cuda():
      with self.test_session():
        result = cuda_op.add_one([5, 4, 3, 2, 1])
        self.assertAllEqual(result.eval(), [6, 5, 4, 3, 2])


if __name__ == '__main__':
  tf.test.main()
