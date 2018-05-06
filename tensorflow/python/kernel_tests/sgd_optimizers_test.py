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

""" Tests for SGD optimizers that use Eigen sqrt/rsqrt function.
    Currently, included ones are Adam, AdaGrad, AdaDelta, and RMSPROP.
    Feel free to add additional ones. These tests are intended for
    validating EIGEN_FAST_MATH build. Some versions of Eigen library
    fail on these tests in AVX512 mode. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.training import adam
from tensorflow.python.training import adagrad
from tensorflow.python.training import adadelta
from tensorflow.python.training import rmsprop

# Input and output dimensions
BATCH_SIZE, IMAGE_DIM, NUM_CLASSES = (2, 10, 10)

# Input data [shape: (BATCH_SIZE, IMAGE_DIM)] and
# output labels [shape: (BATCH_SIZE, NUM_CLASSES)]
images = np.array(
    [[0.48, 0.98, 0.57, 0.47, 0.22, 0.73, 0.08, 0.24, 0.52, 0.38],
     [0.8, 0.78, 0.69, 0.53, 0.38, 0.03, 0.06, 0.23, 0.85, 0.48]],
    dtype=np.float32)
labels = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
                  dtype=np.float32)

# Initial weights [shape: (IMAGE_DIM, NUM_CLASSES)] and
# biases [shape: (NUM_CLASSES,)] for affine transfomation.
W0 = np.array(
    [[-0.37, 0.91, -0.88, -0.35, -0.76, -0.42, 0.25, 0.35, -1.41, 1.9],
     [0.76, -1.36, -0.57, -0.91, -0.56, 0.57, -0.67, 0.52, -0.05, -0.43],
     [-0.82, -0.3, -0.09, -0.36, 0.5, 0.32, 0.96, 0.13, -1.32, -0.13],
     [-0.5, -0.89, 1.23, -1.79, -1.14, -0.53, 0.06, -0.07, -0.6, -1.67],
     [-0.52, 0.04, 0.52, 0.55, 0.46, -2.28, -1.7, 1.55, 0.7, 0.17],
     [0.77, -1.51, 0.02, 0.79, -0.8, -0.55, -1.51, 1.12, -0.84, -2.06],
     [-1.54, -0.8, -0.19, 0.15, 0.07, 2.15, 0.27, 1.79, 0.17, 0.39],
     [-1.62, 0.17, 0.67, 1.26, 0.97, 0.45, -0.95, -0.65, 0.27, 0.03],
     [-0.62, 0.64, 0.02, -0.31, -1.15, -0.37, -0.62, -0.34, 0.75, -1.75],
     [0.03, -0.07, 0.3, 1.06, -1.07, -1.31, -0.96, -0.33, -1.04, -0.04]],
    dtype=np.float32)
b0 = np.array([1.25, 0.98, -0.87, 0.08, 0.38, 1.16, 1.96, -1.21, 0.28, 2.14],
              dtype=np.float32)

# Expected values for loss functions over 100 steps.
expected_loss_list = {
    'adam': np.array(
        [2.4398, 2.351, 2.2646, 2.1806, 2.0991, 2.0204, 1.9444,
         1.8713, 1.8012, 1.734, 1.6697, 1.6083, 1.5497, 1.4939,
         1.4405, 1.3896, 1.3409, 1.2941, 1.2491, 1.2057, 1.1638,
         1.1231, 1.0836, 1.0451, 1.0077, 0.9712, 0.9357, 0.9011,
         0.8676, 0.835, 0.8035, 0.773, 0.7438, 0.7157, 0.6888,
         0.6631, 0.6387, 0.6156, 0.5938, 0.5732, 0.5539, 0.5358,
         0.5188, 0.503, 0.4882, 0.4744, 0.4615, 0.4495, 0.4382,
         0.4277, 0.4178, 0.4085, 0.3998, 0.3915, 0.3836, 0.3761,
         0.369, 0.3621, 0.3555, 0.3492, 0.3431, 0.3373, 0.3316,
         0.3262, 0.3209, 0.3158, 0.3108, 0.3061, 0.3015, 0.297,
         0.2927, 0.2885, 0.2845, 0.2806, 0.2768, 0.2732, 0.2696,
         0.2662, 0.2628, 0.2596, 0.2565, 0.2534, 0.2504, 0.2475,
         0.2447, 0.242, 0.2393, 0.2367, 0.2341, 0.2316, 0.2291,
         0.2267, 0.2243, 0.222, 0.2197, 0.2175, 0.2153, 0.2132,
         0.2111, 0.209], dtype=np.float32),
    'adagrad': np.array(
        [2.4898, 2.4551, 2.4243, 2.3963, 2.3703, 2.3461, 2.3232,
         2.3016, 2.281, 2.2613, 2.2425, 2.2244, 2.2069, 2.1901,
         2.1738, 2.158, 2.1427, 2.1278, 2.1133, 2.0993, 2.0856,
         2.0722, 2.0591, 2.0464, 2.0339, 2.0218, 2.0099, 1.9982,
         1.9868, 1.9756, 1.9646, 1.9538, 1.9432, 1.9329, 1.9227,
         1.9126, 1.9028, 1.8931, 1.8836, 1.8742, 1.865, 1.8559,
         1.847, 1.8382, 1.8295, 1.821, 1.8126, 1.8043, 1.7961,
         1.788, 1.7801, 1.7722, 1.7645, 1.7568, 1.7493, 1.7418,
         1.7345, 1.7272, 1.72, 1.7129, 1.7059, 1.699, 1.6921,
         1.6854, 1.6787, 1.6721, 1.6655, 1.6591, 1.6526, 1.6463,
         1.6401, 1.6339, 1.6277, 1.6216, 1.6156, 1.6097, 1.6038,
         1.598, 1.5922, 1.5865, 1.5808, 1.5752, 1.5696, 1.5641,
         1.5587, 1.5533, 1.5479, 1.5426, 1.5373, 1.5321, 1.527,
         1.5218, 1.5168, 1.5117, 1.5067, 1.5018, 1.4969, 1.492,
         1.4872, 1.4824], dtype=np.float32),
    'adadelta': np.array(
        [2.5306, 2.5306, 2.5305, 2.5305, 2.5304, 2.5304, 2.5303,
         2.5303, 2.5303, 2.5302, 2.5302, 2.5301, 2.5301, 2.53,
         2.53, 2.53, 2.5299, 2.5299, 2.5298, 2.5298, 2.5297,
         2.5297, 2.5296, 2.5296, 2.5296, 2.5295, 2.5295, 2.5294,
         2.5294, 2.5293, 2.5293, 2.5292, 2.5292, 2.5292, 2.5291,
         2.5291, 2.529, 2.529, 2.5289, 2.5289, 2.5288, 2.5288,
         2.5287, 2.5287, 2.5287, 2.5286, 2.5286, 2.5285, 2.5285,
         2.5284, 2.5284, 2.5283, 2.5283, 2.5282, 2.5282, 2.5282,
         2.5281, 2.5281, 2.528, 2.528, 2.5279, 2.5279, 2.5278,
         2.5278, 2.5277, 2.5277, 2.5276, 2.5276, 2.5275, 2.5275,
         2.5275, 2.5274, 2.5274, 2.5273, 2.5273, 2.5272, 2.5272,
         2.5271, 2.5271, 2.527, 2.527, 2.5269, 2.5269, 2.5268,
         2.5268, 2.5267, 2.5267, 2.5267, 2.5266, 2.5266, 2.5265,
         2.5265, 2.5264, 2.5264, 2.5263, 2.5263, 2.5262, 2.5262,
         2.5261, 2.5261], dtype=np.float32),
    'rmsprop': np.array(
        [2.5124, 2.4934, 2.4736, 2.4531, 2.4317, 2.4096, 2.3866,
         2.3629, 2.3383, 2.3129, 2.2866, 2.2596, 2.2318, 2.2033,
         2.1739, 2.1439, 2.1131, 2.0817, 2.0496, 2.017, 1.9837,
         1.95, 1.9157, 1.8811, 1.846, 1.8106, 1.775, 1.7391,
         1.703, 1.6668, 1.6305, 1.5942, 1.5579, 1.5217, 1.4857,
         1.4498, 1.4141, 1.3787, 1.3437, 1.3089, 1.2746, 1.2406,
         1.2071, 1.1741, 1.1415, 1.1095, 1.0781, 1.0472, 1.0168,
         0.9871, 0.958, 0.9295, 0.9016, 0.8744, 0.8478, 0.8219,
         0.7967, 0.7722, 0.7483, 0.7252, 0.7027, 0.681, 0.6599,
         0.6396, 0.62, 0.601, 0.5828, 0.5652, 0.5482, 0.5319,
         0.5163, 0.5013, 0.4868, 0.4729, 0.4596, 0.4469, 0.4346,
         0.4229, 0.4116, 0.4007, 0.3904, 0.3804, 0.3708, 0.3616,
         0.3528, 0.3443, 0.3361, 0.3283, 0.3207, 0.3134, 0.3064,
         0.2997, 0.2931, 0.2868, 0.2808, 0.2749, 0.2692, 0.2637,
         0.2584, 0.2533], dtype=np.float32)
}

# Defining a graph that will be shared by all optimizers
graph = ops.Graph()
with graph.as_default():
  # A logistic regression model
  x = constant_op.constant(images, dtype=dtypes.float32,
                           shape=[BATCH_SIZE, IMAGE_DIM], name="x")
  W = variables.Variable(W0, dtype=dtypes.float32, name='weight')
  b = variables.Variable(b0, dtype=dtypes.float32, name='bias')
  y = math_ops.matmul(x, W) + b
  y_ = constant_op.constant(labels, dtype=dtypes.float32,
                            shape=[BATCH_SIZE, NUM_CLASSES], name="y")

  # Loss function
  cross_entropy = math_ops.reduce_mean(
      nn_ops.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  # A dictionary collecting one training step for different optimizers
  train_step_list = {
      'adam': adam.AdamOptimizer(
          learning_rate=0.01, beta1=0.9, beta2=0.999,
          epsilon=1e-08, use_locking=False,
          name='Adam').minimize(cross_entropy),
      'adagrad': adagrad.AdagradOptimizer(
          learning_rate=0.01).minimize(cross_entropy),
      'adadelta': adadelta.AdadeltaOptimizer(
          learning_rate=0.01).minimize(cross_entropy),
      'rmsprop': rmsprop.RMSPropOptimizer(
          learning_rate=0.01).minimize(cross_entropy)
  }
  initializer = variables.global_variables_initializer()


class SGDOptimizerTest(test.TestCase):
  """A set op tests that use sqrt/rsqrt functions from Eigen
    library with SIMD instructions.
  """
  def _testSGDOptimizer(self, optimizer_name): # pylint: disable=invalid-name
    with self.test_session(graph=graph) as sess:
      train_step = train_step_list[optimizer_name]
      expected_loss = expected_loss_list[optimizer_name]
      actual_loss = np.array([], dtype=np.float32)
      sess.run(initializer)
      for _ in range(expected_loss.size):
        sess.run(train_step)
        actual_loss = np.append(actual_loss, sess.run(cross_entropy))
      self.assertAllClose(expected_loss, actual_loss, rtol=1e-4, atol=1e-4)

  def testAdamOptimizers(self): # pylint: disable=invalid-name
    print("Testing Adam Optimizer....")
    self._testSGDOptimizer('adam')

  def testAdaGradOptimizers(self): # pylint: disable=invalid-name
    print("Testing AdaGrad Optimizer....")
    self._testSGDOptimizer('adagrad')

  def testAdaDeltaOptimizers(self): # pylint: disable=invalid-name
    print("Testing AdaDelta Optimizer....")
    self._testSGDOptimizer('adadelta')

  def testRMSPROPOptimizers(self): # pylint: disable=invalid-name
    print("Testing RMSPROP Optimizer....")
    self._testSGDOptimizer('rmsprop')

if __name__ == '__main__':
  test.main()
