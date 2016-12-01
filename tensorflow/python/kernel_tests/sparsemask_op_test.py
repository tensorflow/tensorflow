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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SparseMaskTest(tf.test.TestCase):

  def testBasic(self):
    values = np.random.rand(4, 4).astype(np.single)
    indices = np.array([0, 2, 3, 4], dtype=np.int32)
    mask_indices = np.array([0], dtype=np.int32)

    out_values = values[1:, :]
    out_indices = np.array([2, 3, 4], dtype=np.int32)

    with self.test_session() as sess:
      values_tensor = tf.convert_to_tensor(values)
      indices_tensor = tf.convert_to_tensor(indices)
      mask_indices_tensor = tf.convert_to_tensor(mask_indices)

      t = tf.IndexedSlices(values_tensor, indices_tensor)
      masked_t = tf.sparse_mask(t, mask_indices_tensor)

      tf_out_values, tf_out_indices = sess.run([masked_t.values,
                                                masked_t.indices])

      self.assertAllEqual(tf_out_values, out_values)
      self.assertAllEqual(tf_out_indices, out_indices)

if __name__ == "__main__":
  tf.test.main()
