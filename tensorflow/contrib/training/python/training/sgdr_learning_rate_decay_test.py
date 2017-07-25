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

"""Functional test for sgdr learning rate decay."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from sgdr_learning_rate_decay import sgdr_decay
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import dtypes


class SGDRDecayTest(test_util.TensorFlowTestCase):

  # Original iterative approach
  # https://github.com/loshchil/SGDR/blob/master/SGDR_WRNs.py
  def get_original_values(self, lr, Te, multFactor, iter_per_epoch, epochs):
    t0 = math.pi / 2.0
    tt = 0
    TeNext = Te

    lr_values = []
    sh_lr = lr
    for epoch in range(epochs):
      for iteration in range(iter_per_epoch):
        # In the original approach training function is executed here
        lr_values.append(sh_lr)

        dt = 2.0 * math.pi / float(2.0 * Te)
        tt = tt + float(dt) / iter_per_epoch
        if tt >= math.pi:
          tt = tt - math.pi
        curT = t0 + tt
        new_lr = lr * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
        sh_lr = new_lr
      if (epoch + 1) == TeNext:  # time to restart
        sh_lr = lr
        tt = 0                # by setting to 0 we set lr to lr_max, see above
        Te = Te * multFactor  # change the period of restarts
        TeNext = TeNext + Te  # note the next restart's epoch

    return lr_values

  def get_sgdr_values(self, lr, Te, multFactor, iters):
    step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
                                   name="step", container="", shared_name="")
    state_ops.assign(step, 0).op.run()

    decay = sgdr_decay(lr, step, Te, multFactor)
    lr_values = []
    for i in range(iters):
      lr_values.append(decay.eval())
      state_ops.assign(step, step + 1).op.run()

    return lr_values

  def testCompareToOriginal(self):
    with self.test_session():
      lr = 10.0
      t_e = 2
      t_mul = 3
      iters = 10
      epochs = 50

      org_lr = self.get_original_values(lr, t_e, t_mul, iters, epochs)
      sgdr_lr = self.get_sgdr_values(lr, t_e*iters, t_mul, iters*epochs)

      for org, sgdr in zip(org_lr, sgdr_lr):
        self.assertAllClose(org, sgdr)

  def testMDecay(self):
    with self.test_session():
      step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
                                     name="step", container="", shared_name="")
      state_ops.assign(step, 0).op.run()

      lr = 0.1
      t_e = 10
      t_mul = 3
      m_mul = 0.9

      decay = sgdr_decay(lr, step, t_e, t_mul, m_mul)

      self.assertAllClose(decay.eval(), lr)

      state_ops.assign(step, t_e).op.run()
      self.assertAllClose(decay.eval(), lr * m_mul)

      state_ops.assign(step, t_e + t_e*t_mul).op.run()
      self.assertAllClose(decay.eval(), lr * m_mul**2)

      state_ops.assign(step, t_e + t_e*t_mul + t_e * (t_mul**2)).op.run()
      self.assertAllClose(decay.eval(), lr * (m_mul**3))

  def testCos(self):
    with self.test_session():
      step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
                                     name="step", container="", shared_name="")
      state_ops.assign(step, 0).op.run()
      lr = 0.2
      t_e = 1000
      t_mul = 1

      decay = sgdr_decay(lr, step, t_e, t_mul)

      self.assertAllClose(decay.eval(), lr)

      state_ops.assign(step, t_e//2).op.run()
      self.assertAllClose(decay.eval(), lr/2)

      state_ops.assign(step, t_e).op.run()
      self.assertAllClose(decay.eval(), lr)

      state_ops.assign(step, t_e*3//2).op.run()
      self.assertAllClose(decay.eval(), lr/2)

if __name__ == "__main__":
  googletest.main()
