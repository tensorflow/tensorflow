# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0. Licensed to the Apache
# Software Foundation. You may not use this file except in compliance with the
# License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for Layer-wise Adaptive Rate Scaling optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import lars_optimizer as lo
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class LARSOptimizerTest(test.TestCase):

  def testLARSGradientOneStep(self):
    for _ in range(10):
      for dtype in [dtypes.float32, dtypes.float64]:
        with self.cached_session() as sess:
          shape = [3, 3]
          var_np = np.ones(shape)
          grad_np = np.ones(shape)
          lr_np = 0.1
          m_np = 0.9
          wd_np = 0.1
          ep_np = 1e-5
          eeta = 0.1
          vel_np = np.zeros(shape)

          var = variables.Variable(var_np, dtype=dtype)
          grad = variables.Variable(grad_np, dtype=dtype)
          opt = lo.LARSOptimizer(
              learning_rate=lr_np,
              momentum=m_np,
              weight_decay=wd_np,
              eeta=eeta,
              epsilon=ep_np)

          step = opt.apply_gradients([(grad, var)])
          variables.global_variables_initializer().run()

          pre_var = sess.run(var)
          pre_vel = sess.run(opt.get_slot(var, 'momentum'))
          self.assertAllClose(var_np, pre_var)
          self.assertAllClose(vel_np, pre_vel)

          step.run()
          post_var = sess.run(var)
          post_vel = sess.run(opt.get_slot(var, 'momentum'))

          w_norm = np.linalg.norm(var_np.flatten(), ord=2)
          g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
          trust_ratio = eeta * w_norm / (g_norm + wd_np * w_norm + ep_np)
          scaled_lr = lr_np * trust_ratio

          vel_np = m_np * vel_np + grad_np
          var_np -= scaled_lr * vel_np

          self.assertAllClose(var_np, post_var)
          self.assertAllClose(vel_np, post_vel)

  def testLARSGradientMultiStep(self):
    for _ in range(10):
      for dtype in [dtypes.float32, dtypes.float64]:
        with self.cached_session() as sess:
          shape = [3, 3]
          var_np = np.ones(shape)
          grad_np = np.ones(shape)
          lr_np = 0.1
          m_np = 0.9
          wd_np = 0.1
          ep_np = 1e-5
          eeta = 0.1
          vel_np = np.zeros(shape)

          var = variables.Variable(var_np, dtype=dtype)
          grad = variables.Variable(grad_np, dtype=dtype)
          opt = lo.LARSOptimizer(
              learning_rate=lr_np,
              momentum=m_np,
              eeta=eeta,
              weight_decay=wd_np,
              epsilon=ep_np)

          step = opt.apply_gradients([(grad, var)])
          variables.global_variables_initializer().run()

          pre_var = sess.run(var)
          pre_vel = sess.run(opt.get_slot(var, 'momentum'))
          self.assertAllClose(var_np, pre_var)
          self.assertAllClose(vel_np, pre_vel)

          for _ in range(10):
            step.run()

            post_var = sess.run(var)
            post_vel = sess.run(opt.get_slot(var, 'momentum'))

            w_norm = np.linalg.norm(var_np.flatten(), ord=2)
            g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
            trust_ratio = eeta * w_norm / (g_norm + wd_np * w_norm + ep_np)
            scaled_lr = lr_np * trust_ratio

            vel_np = m_np * vel_np + grad_np
            var_np -= scaled_lr * vel_np

            self.assertAllClose(var_np, post_var)
            self.assertAllClose(vel_np, post_vel)


if __name__ == '__main__':
  test.main()
