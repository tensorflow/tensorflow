# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for python single_image_random_dot_stereograms_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.image.python.ops.single_image_random_dot_stereograms \
    import single_image_random_dot_stereograms
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class SingleImageRandomDotStereogramsTest(test_util.TensorFlowTestCase):

  def test_shape_function_default(self):
    """
    NOTE: The output_image_shape is [X, Y, C]
    while the output data is [Y, X, C] (or [H, W, C]).
    As a result, by default the output_image_shape has the value
    of [1024, 768, 1], but the output data will be [768, 1024, 1].
    """
    x_np = [[1, 2, 3, 3, 2, 1],
            [1, 2, 3, 4, 5, 2],
            [1, 2, 3, 4, 5, 3],
            [1, 2, 3, 4, 5, 4],
            [6, 5, 4, 4, 5, 5]]
    x_tf = constant_op.constant(x_np)
    # By default [1024, 768, 1] => [768, 1024, 1].
    sirds_1 = single_image_random_dot_stereograms(
        x_tf,
        convergence_dots_size=8,
        number_colors=256,
        normalize=True)
    shape_1 = sirds_1.get_shape().as_list()
    self.assertEqual(shape_1, [768, 1024, 1])
    with self.test_session():
      r_tf_1 = sirds_1.eval()
      self.assertAllEqual(shape_1, r_tf_1.shape)

    # If color > 256 then [1024, 768, 3] => [768, 1024, 3].
    sirds_2 = single_image_random_dot_stereograms(
        x_tf,
        convergence_dots_size=8,
        number_colors=512,
        normalize=True)
    shape_2 = sirds_2.get_shape().as_list()
    self.assertEqual(shape_2, [768, 1024, 3])
    with self.test_session():
      r_tf_2 = sirds_2.eval()
      self.assertAllEqual(shape_2, r_tf_2.shape)

    # If explicitly set output_image_shape to [1200, 800, 1],
    # then the output data should be [800, 1200, 1].
    sirds_3 = single_image_random_dot_stereograms(
        x_tf,
        convergence_dots_size=8,
        number_colors=256,
        normalize=True,
        output_image_shape=[1200, 800, 1])
    shape_3 = sirds_3.get_shape().as_list()
    self.assertEqual(shape_3, [800, 1200, 1])
    with self.test_session():
      r_tf_3 = sirds_3.eval()
      self.assertAllEqual(shape_3, r_tf_3.shape)


if __name__ == '__main__':
  googletest.main()
