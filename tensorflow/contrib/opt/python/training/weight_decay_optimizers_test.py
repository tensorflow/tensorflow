# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for optimizers with weight decay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.contrib.opt.python.training import weight_decay_optimizers

WEIGHT_DECAY = 0.01


def adamw_update_numpy(param, g_t, t, m, v, lr=0.001, beta1=0.9,
                       beta2=0.999, epsilon=1e-8):
  lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = (param - lr_t * m_t / (np.sqrt(v_t) + epsilon) -
             (param * WEIGHT_DECAY))
  return param_t, m_t, v_t


def momentumw_update_numpy(param, g_t, t, m, v, lr=0.001, momentum=0.9):
  # v, t are not needed for momentum optimizer
  m = momentum * m + g_t
  param_t = param - lr * m - param * WEIGHT_DECAY
  return param_t, m, None


class WeightDecayOptimizerTest(test.TestCase):

  def doTest(self, optimizer, update_fn, optimizer_name, slot_name,
             use_resource=False, do_sparse=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.test_session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)

        if do_sparse:
          grads0_np_indices = np.array([0, 1], dtype=np.int32)
          grads0 = ops.IndexedSlices(constant_op.constant(grads0_np),
                                     constant_op.constant(grads0_np_indices),
                                     constant_op.constant([2]))
          grads1_np_indices = np.array([0, 1], dtype=np.int32)
          grads1 = ops.IndexedSlices(constant_op.constant(grads1_np),
                                     constant_op.constant(grads1_np_indices),
                                     constant_op.constant([2]))
        else:
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)

        opt = optimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))


        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of the optimizer
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          var0_np, m0, v0 = update_fn(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = update_fn(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
          if use_resource:
            self.assertEqual("var0_%d/%s:0" % (i, optimizer_name),
                             opt.get_slot(var=var0, name=slot_name).name)


class AdamWOptimizerTest(WeightDecayOptimizerTest):

  @staticmethod
  def get_optimizer():
    return weight_decay_optimizers.AdamWOptimizer(WEIGHT_DECAY)

  def testSparse(self):
    self.doTest(self.get_optimizer, adamw_update_numpy, "AdamW", "m",
                use_resource=False, do_sparse=True)

  def testResourceSparse(self):
    self.doTest(self.get_optimizer, adamw_update_numpy, "AdamW", "m",
                use_resource=True, do_sparse=True)

  def testBasic(self):
    self.doTest(self.get_optimizer, adamw_update_numpy, "AdamW", "m",
                use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTest(self.get_optimizer, adamw_update_numpy, "AdamW", "m",
                use_resource=True)


class MomentumWOptimizerTest(WeightDecayOptimizerTest):

  @staticmethod
  def get_optimizer():
    return weight_decay_optimizers.MomentumWOptimizer(WEIGHT_DECAY, 0.001, 0.9)

  def testSparse(self):
    self.doTest(self.get_optimizer, momentumw_update_numpy, "MomentumW",
                "momentum", use_resource=False, do_sparse=True)

  def testResourceSparse(self):
    self.doTest(self.get_optimizer, momentumw_update_numpy, "MomentumW",
                "momentum", use_resource=True, do_sparse=True)

  def testBasic(self):
    self.doTest(self.get_optimizer, momentumw_update_numpy, "MomentumW",
                "momentum", use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTest(self.get_optimizer, momentumw_update_numpy, "MomentumW",
                "momentum", use_resource=True)


class ExtendWithWeightDecayTest(WeightDecayOptimizerTest):

  @staticmethod
  def get_optimizer():
    AdamW = weight_decay_optimizers.extend_with_decoupled_weight_decay(
        adam.AdamOptimizer)
    return AdamW(WEIGHT_DECAY)

  def testBasic(self):
    self.doTest(self.get_optimizer, adamw_update_numpy, "Adam", "m",
                use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTest(self.get_optimizer, adamw_update_numpy, "Adam", "m",
                use_resource=True)



if __name__ == "__main__":
  test.main()
