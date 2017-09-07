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
"""Utils for writing tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class Base(test.TestCase):
  """A class with some useful methods for testing."""

  def eval(self, tensors):
    with self.test_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)

      try:
        results = sess.run(tensors)
      finally:
        coord.request_stop()
        coord.join(threads)

      return results

  def assertTensorsEqual(self, tensor_0, tensor_1):
    [tensor_0_eval, tensor_1_eval] = self.eval([tensor_0, tensor_1])
    self.assertAllEqual(tensor_0_eval, tensor_1_eval)

  def assertLabeledTensorsEqual(self, tensor_0, tensor_1):
    self.assertEqual(tensor_0.axes, tensor_1.axes)
    self.assertTensorsEqual(tensor_0.tensor, tensor_1.tensor)
