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

"""Tests for batch_norm related functionality in tensorflow.ops.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.cudnn_batchnorm import batch_norm_training
from tensorflow.contrib.cudnn_batchnorm import load_cudnn_batchnorm_so

class CudnnBatchNormTest(tf.test.TestCase):
    def _npBatchNorm(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization, shift_after_normalization):
        y = (x - m) / np.sqrt(v + epsilon)
        y = y * gamma if scale_after_normalization else y
        return y + beta if shift_after_normalization else y

    def testBatchNormTraining(self):
      """Test BatchNormTraining ops
      """
      load_cudnn_batchnorm_so.Load()
      x_shape = [3, 5, 4, 2]
      param_shape = [5]
      x_val = np.random.random_sample(x_shape).astype(np.float32)
      beta_val = np.random.random_sample(param_shape).astype(np.float32)
      gamma_val = np.random.random_sample(param_shape).astype(np.float32)
      # TODO check the non gpu case
      use_gpu = True
      with self.test_session(use_gpu=use_gpu) as sess:
        # TODO support other arguments
        scale_after_normalization = True
        shift_after_normalization = True
        x = tf.constant(x_val, name="x")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        epsilon = 0.001
        # TODO check / implement shift after scale after
        bn = batch_norm_training(x, beta, gamma, epsilon, data_format="NCHW")
        m_val = np.mean(x_val, axis=(0, 2, 3), keepdims=True)
        v_val = np.var(x_val, axis=(0, 2, 3), keepdims=True)
        beta_reshape = beta_val.reshape([1, param_shape[0], 1, 1])
        gamma_reshape = gamma_val.reshape([1, param_shape[0], 1, 1])
        np_bn = self._npBatchNorm(
          x_val, m_val, v_val, beta_reshape, gamma_reshape, epsilon,
          scale_after_normalization, shift_after_normalization)

        bn_val = sess.run(bn)
        self.assertAllClose(bn_val, np_bn, atol=1e-4, rtol=1e-4)

        # Test NHWC
        x_T = tf.transpose(x, (0, 2, 3, 1))
        bn = batch_norm_training(x_T, beta, gamma, epsilon, data_format="NHWC")
        bn_val = sess.run(bn)
        self.assertAllClose(bn_val, np_bn.transpose(0, 2, 3, 1), atol=1e-4, rtol=1e-4)

    def testBatchNormTrainingGrad(self):
      x_shape = [3, 5, 4, 6]
      param_shape = [5]
      np.random.seed(1)  # Make it reproducible.
      x_val = np.random.random_sample(x_shape).astype(np.float32)
      beta_val = np.random.random_sample(param_shape).astype(np.float32)
      gamma_val = np.random.random_sample(param_shape).astype(np.float32)
      with self.test_session():
        x = tf.constant(x_val, name="x")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        epsilon = 0.001

        output = batch_norm_training(x, beta, gamma, variance_epsilon=epsilon, data_format="NCHW")
        all_params = [x, beta, gamma]
        all_shapes = [x_shape, param_shape, param_shape]
        for param_index in range(len(all_params)):
          err = tf.test.compute_gradient_error(
                all_params[param_index], all_shapes[param_index], output, x_shape)
          self.assertLess(err, 1e-3)

        # Test NHWC
        x_T = tf.transpose(x, (0, 2, 3, 1))
        x_T_shape = [x_shape[0], x_shape[2], x_shape[3], x_shape[1]]
        output = batch_norm_training(x_T, beta, gamma, variance_epsilon=epsilon, data_format="NHWC")
        all_params = [x_T, beta, gamma]
        all_shapes = [x_T_shape, param_shape, param_shape]
        for param_index in range(len(all_params)):
          err = tf.test.compute_gradient_error(
                all_params[param_index], all_shapes[param_index], output, x_T_shape)
          self.assertLess(err, 1e-3)

CudnnBatchNormTest("testBatchNormTraining").testBatchNormTraining()
