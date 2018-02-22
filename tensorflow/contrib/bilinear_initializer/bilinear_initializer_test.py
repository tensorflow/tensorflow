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
# =============================================================================
"""Tests for third_party.tensorflow.contrib.bilinear_initializer_op."""

import numpy as np
import skimage.transform as sktr
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.contrib.bilinear_initializer.bilinear_initializer_op import \
  bilinear_initializer


class BilinearInitializerTest(test.TestCase):

  def testBilinearInitializer(self):
    with self.test_session() as sess:
      # Define parameters.
      height, width, channel = 3, 3, 3
      factor = 2
      kernel_width = 2 * factor - factor % 2
      num_output = 3

      # Generate input image.
      img = self.generate_input_img(height, width, channel)

      # Generate benchmark image (ski).
      img_benchmark = self.ski_upsample(factor, img)

      # Generate test image (tensorflow).
      dims = [kernel_width, kernel_width, channel, num_output]
      tf_filter = bilinear_initializer(dims).eval()
      new_h = factor * height
      new_w = factor * width
      img_test = self.tf_upsample(sess,
                                  tf_filter,
                                  new_h,
                                  new_w,
                                  channel,
                                  factor,
                                  img)
      self.assertTrue(np.allclose(img_benchmark, img_test))

  def generate_input_img(self, height, width, channel):
    """Generate input image with a given height, width, and number of channels.
    """
    x, y = np.ogrid[:height, :width]
    img = np.repeat((x + y)[..., np.newaxis], channel, 2)
    return img / float(height * width)

  def ski_upsample(self, factor, input_img):
    """Benchmark for testing. Use skikit learn library.
       order = 1 means bilinear initializer
    """
    return sktr.rescale(input_img, factor, mode='constant', cval=0, order=1)

  def tf_upsample(self, sess, tf_filter, new_h, new_w, channel, factor,
                  input_img):
    expanded_img = np.expand_dims(input_img, axis=0)
    res = nn_ops.conv2d_transpose(
        expanded_img,
        tf_filter,
        output_shape=[1, new_h, new_w, channel],
        strides=[1, factor, factor, 1])
    result = sess.run(res)
    return result.squeeze()


if __name__ == "__main__":
  test.main()
