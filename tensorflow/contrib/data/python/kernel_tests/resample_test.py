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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.contrib.data.python.ops import resampling
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.util import compat


class ResampleTest(test.TestCase):

  def testInitialKnownDistribution(self):
    self._testDistribution(initial_known=True)

  def testInitialNotKnownDistribution(self):
    self._testDistribution(initial_known=False)

  def _testDistribution(self, initial_known):
    classes = np.random.randint(5, size=(20000,))  # Uniformly sampled
    target_dist = [0.9, 0.05, 0.05, 0.0, 0.0]
    initial_dist = [0.2] * 5 if initial_known else None
    iterator = (dataset_ops.Dataset.from_tensor_slices(classes).shuffle(
        200, seed=21).map(lambda c: (c, string_ops.as_string(c))).apply(
            resampling.rejection_resample(
                target_dist=target_dist,
                initial_dist=initial_dist,
                class_func=lambda c, _: c,
                seed=27)).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()
    variable_init_op = variables.local_variables_initializer()

    with self.test_session() as sess:
      sess.run(variable_init_op)
      sess.run(init_op)
      returned = []
      with self.assertRaises(errors.OutOfRangeError):
        while True:
          returned.append(sess.run(get_next))

    returned_classes, returned_classes_and_data = zip(*returned)
    _, returned_data = zip(*returned_classes_and_data)
    self.assertAllEqual([compat.as_bytes(str(c))
                         for c in returned_classes], returned_data)
    total_returned = len(returned_classes)
    # Subsampling rejects a large percentage of the initial data in
    # this case.
    self.assertGreater(total_returned, 20000 * 0.2)
    class_counts = np.array([
        len([True for v in returned_classes if v == c])
        for c in range(5)])
    returned_dist = class_counts / total_returned
    self.assertAllClose(target_dist, returned_dist, atol=1e-2)

  def testVariableDevicePlacement(self):
    classes = np.random.randint(5, size=(20000,))  # Uniformly sampled
    target_dist = [0.9, 0.05, 0.05, 0.0, 0.0]
    with ops.device(
        device_setter.replica_device_setter(ps_tasks=1, ps_device="/cpu:0")):
      _ = (dataset_ops.Dataset.from_tensor_slices(classes).shuffle(
          200, seed=21).map(lambda c: (c, string_ops.as_string(c))).apply(
              resampling.rejection_resample(
                  target_dist=target_dist,
                  initial_dist=None,
                  class_func=lambda c, _: c,
                  seed=27)))

      self.assertEqual(1, len(variables.local_variables()))
      self.assertEqual(b"",
                       compat.as_bytes(variables.local_variables()[0].device))


if __name__ == "__main__":
  test.main()
