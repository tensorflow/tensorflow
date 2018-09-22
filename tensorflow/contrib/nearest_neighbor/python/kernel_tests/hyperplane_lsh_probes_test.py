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

"""Tests for hyperplane_lsh_probes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.nearest_neighbor.python.ops.nearest_neighbor_ops import hyperplane_lsh_probes
from tensorflow.python.platform import test


class HyperplaneLshProbesTest(test.TestCase):

  # We only test the batch functionality of the op here because the multiprobe
  # tests in hyperplane_lsh_probes_test.cc already cover most of the LSH
  # functionality.
  def simple_batch_test(self):
    with self.cached_session():
      hyperplanes = np.eye(4)
      points = np.array([[1.2, 0.5, -0.9, -1.0], [2.0, -3.0, 1.0, -1.5]])
      product = np.dot(points, hyperplanes)
      num_tables = 2
      num_hyperplanes_per_table = 2
      num_probes = 4
      hashes, tables = hyperplane_lsh_probes(product,
                                             num_tables,
                                             num_hyperplanes_per_table,
                                             num_probes)

      self.assertAllEqual(hashes.eval(), [[3, 0, 2, 2], [2, 2, 0, 3]])
      self.assertAllEqual(tables.eval(), [[0, 1, 0, 1], [0, 1, 1, 1]])


if __name__ == '__main__':
  test.main()
