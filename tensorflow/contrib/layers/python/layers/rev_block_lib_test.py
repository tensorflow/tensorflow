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
"""Tests for RevBlock."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import rev_block_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class RevBlockTest(test.TestCase):
  CHANNELS = 8
  NUM_LAYERS = 4
  BATCH_SIZE = 16

  def testForwardBackward(self):

    def f(x):
      return core_layers.dense(x, self.CHANNELS // 2, use_bias=True)

    def g(x):
      return core_layers.dense(x, self.CHANNELS // 2, use_bias=True)

    x = random_ops.random_uniform(
        [self.BATCH_SIZE, self.CHANNELS], dtype=dtypes.float32)
    x1, x2 = array_ops.split(x, 2, axis=-1)

    block = rev_block_lib.RevBlock(f, g, num_layers=3)
    y1, y2 = block.forward(x1, x2)
    x1_inv, x2_inv = block.backward(y1, y2)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      x1, x2, x1_inv, x2_inv = sess.run([x1, x2, x1_inv, x2_inv])

      self.assertAllClose(x1, x1_inv, atol=1e-5)
      self.assertAllClose(x2, x2_inv, atol=1e-5)

  def testBackwardForward(self):

    def f(x):
      return core_layers.dense(x, self.CHANNELS // 2, use_bias=True)

    def g(x):
      return core_layers.dense(x, self.CHANNELS // 2, use_bias=True)

    y = random_ops.random_uniform(
        [self.BATCH_SIZE, self.CHANNELS], dtype=dtypes.float32)
    y1, y2 = array_ops.split(y, 2, axis=-1)

    block = rev_block_lib.RevBlock(f, g, num_layers=3)
    x1, x2 = block.backward(y1, y2)
    y1_inv, y2_inv = block.forward(x1, x2)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      y1, y2, y1_inv, y2_inv = sess.run([y1, y2, y1_inv, y2_inv])

      self.assertAllClose(y1, y1_inv, rtol=1e-5)
      self.assertAllClose(y2, y2_inv, rtol=1e-5)

  def _testRevBlock(self,
                    x=None,
                    f=None,
                    g=None,
                    f_side_input=None,
                    g_side_input=None):
    random_seed.set_random_seed(1234)

    if f is None:

      def f(x):  # pylint: disable=function-redefined
        return core_layers.dense(x, self.CHANNELS // 2, use_bias=True)

    if g is None:

      def g(x):  # pylint: disable=function-redefined
        return core_layers.dense(x, self.CHANNELS // 2, use_bias=True)

    if f_side_input is None:
      f_side_input = []

    if g_side_input is None:
      g_side_input = []

    if x is None:
      x = random_ops.random_uniform(
          [self.BATCH_SIZE, self.CHANNELS], dtype=dtypes.float32)
    x1, x2 = array_ops.split(x, 2, axis=-1)

    with variable_scope.variable_scope("rev_test") as vs:
      y1_rev, y2_rev = rev_block_lib.rev_block(
          x1,
          x2,
          f,
          g,
          f_side_input=f_side_input,
          g_side_input=g_side_input,
          num_layers=self.NUM_LAYERS)
      y_rev = array_ops.concat([y1_rev, y2_rev], axis=1)
      fg_vars = vs.trainable_variables()

    num_vars = len(variables.global_variables())
    with variable_scope.variable_scope(vs, reuse=True):
      y1, y2 = rev_block_lib.rev_block(
          x1,
          x2,
          f,
          g,
          f_side_input=f_side_input,
          g_side_input=g_side_input,
          num_layers=self.NUM_LAYERS,
          is_training=False)
      y = array_ops.concat([y1, y2], axis=1)
    # Ensure no new vars were created - full reuse
    assert len(variables.global_variables()) == num_vars

    loss_rev = math_ops.reduce_mean(y_rev + 10.)
    loss = math_ops.reduce_mean(y + 10.)

    wrt = [x] + f_side_input + g_side_input + fg_vars
    grads_rev = gradients_impl.gradients(loss_rev, wrt)
    grads = gradients_impl.gradients(loss, wrt)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      y_val, yd_val, gd_val, g_val = sess.run([y, y_rev, grads_rev, grads])
      self.assertAllClose(y_val, yd_val)
      for g1, g2 in zip(gd_val, g_val):
        self.assertAllClose(g1, g2, rtol=1e-5)

  def testRevBlock(self):
    self._testRevBlock()

  def testSideInput(self):
    f_side_input = random_ops.random_uniform(
        [self.BATCH_SIZE, self.CHANNELS // 2])

    def f(x, side_input):
      return core_layers.dense(
          x, self.CHANNELS // 2, use_bias=True) + side_input[0]

    self._testRevBlock(f=f, f_side_input=[f_side_input])

  def testMultipleFns(self):

    def f1(x):
      return core_layers.dense(x, self.CHANNELS // 2)

    def f2(x):
      return core_layers.dense(x, self.CHANNELS // 2, activation=nn_ops.relu)

    self._testRevBlock(f=[f1, f2, f1, f2])

  def testConvAndBatchNorm(self):

    x = random_ops.random_uniform(
        [self.BATCH_SIZE, 10, self.CHANNELS], dtype=dtypes.float32)

    def f(x):
      x = convolutional.conv1d(x, self.CHANNELS // 2, 3, padding="same")
      x = layers.batch_norm(x, is_training=False)
      x = convolutional.conv1d(x, self.CHANNELS // 2, 3, padding="same")
      x = layers.batch_norm(x, is_training=False)
      return x

    self._testRevBlock(x=x, f=f)

  def testReuse(self):

    def f(x):
      return core_layers.dense(x, self.CHANNELS // 2)

    def g(x):
      return core_layers.dense(x, self.CHANNELS // 2)

    x = random_ops.random_uniform(
        [self.BATCH_SIZE, self.CHANNELS], dtype=dtypes.float32)
    x1, x2 = array_ops.split(x, 2, axis=-1)

    with variable_scope.variable_scope("test"):
      y1, y2 = rev_block_lib.rev_block(x1, x2, f, g, num_layers=self.NUM_LAYERS)

    num_vars_before = len(variables.global_variables())

    with variable_scope.variable_scope("test", reuse=True):
      y1, y2 = rev_block_lib.rev_block(x1, x2, f, g, num_layers=self.NUM_LAYERS)

    num_vars_after = len(variables.global_variables())
    self.assertEqual(num_vars_before, num_vars_after)

    loss = math_ops.reduce_mean(y1 + y2)
    _ = gradients_impl.gradients(loss,
                                 [x] + variables.trainable_variables())

    with variable_scope.variable_scope("test", reuse=True):
      y1, y2 = rev_block_lib.rev_block(x1, x2, f, g, num_layers=self.NUM_LAYERS)

    num_vars_after = len(variables.global_variables())
    self.assertEqual(num_vars_before, num_vars_after)


class RecomputeTest(test.TestCase):

  def testRecompute(self):

    def layer(x, name=None):
      with variable_scope.variable_scope(name, default_name="layer"):
        x = layers.layer_norm(x)
        x = convolutional.conv1d(
            x,
            10,
            1,
            use_bias=False,
            kernel_initializer=init_ops.constant_initializer(42.42))
        x = nn_ops.relu(x)
        return x

    def fn(x):
      out = x
      for _ in range(3):
        out = layer(out)
      return out

    @rev_block_lib.recompute_grad
    def fn_recompute(x):
      return fn(x)

    @rev_block_lib.recompute_grad(use_data_dep=True)
    def fn_use_data_dep(x):
      return fn(x)

    @rev_block_lib.recompute_grad(tupleize_grads=True)
    def fn_tupleize(x):
      return fn(x)

    @rev_block_lib.recompute_grad(use_data_dep=True, tupleize_grads=True)
    def fn_both(x):
      return fn(x)

    x = random_ops.random_uniform((3, 1, 3))

    names_and_fns = [
        ("recompute", fn_recompute),
        ("regular", fn),
        ("use_data_dep", fn_use_data_dep),
        ("tupleize", fn_tupleize),
        ("tuple_and_data_dep", fn_both),
    ]
    outputs_and_vars = []
    for name, wrapped_fn in names_and_fns:
      with variable_scope.variable_scope(name, use_resource=True) as vs:
        out = math_ops.reduce_sum(wrapped_fn(x))
        outputs_and_vars.append((out, vs.trainable_variables()))

    all_grads = []
    for out, scope_vars in outputs_and_vars:
      all_grads.append(gradients_impl.gradients(out, scope_vars))

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs = list(zip(*outputs_and_vars))[0]
      outs, all_grads_val = sess.run([outputs, all_grads])

      # All outputs are the same
      current = outs[0]
      for out in outs[1:]:
        self.assertAllClose(current, out)
        current = out

      # All gradients are the same
      for grads in zip(all_grads_val):
        current = grads[0]
        for g in grads[1:]:
          self.assertAllClose(current, g)
          current = g

  def testDoubleCallInSameScopeFails(self):

    @rev_block_lib.recompute_grad
    def layer_with_recompute(inputs):
      return core_layers.dense(inputs, 2)

    with variable_scope.variable_scope("layer", use_resource=True):
      inputs = array_ops.ones((2, 4), dtypes.float32)
      out1 = layer_with_recompute(inputs)
      out2 = layer_with_recompute(inputs) + out1
      out = math_ops.reduce_sum(out2)

    tvars = variables.trainable_variables()
    assert len(tvars) == 4
    with self.assertRaisesWithPredicateMatch(
        ValueError, "called twice in the same enclosing scope"):
      gradients_impl.gradients(out, [inputs] + tvars)

  def testDoubleCallInUniqueScope(self):

    @rev_block_lib.recompute_grad
    def layer_with_recompute(inputs):
      with variable_scope.variable_scope("inner", use_resource=True):
        return core_layers.dense(inputs, 2)

    with variable_scope.variable_scope("layer", use_resource=True):
      inputs = array_ops.ones((2, 4), dtypes.float32)

      with variable_scope.variable_scope("layer1", use_resource=True):
        out1 = layer_with_recompute(inputs)
      with variable_scope.variable_scope("layer2", use_resource=True):
        out2 = layer_with_recompute(inputs) + out1
      out = math_ops.reduce_sum(out2)

    tvars = variables.trainable_variables()
    assert len(tvars) == 4
    grads = gradients_impl.gradients(out, [inputs] + tvars)
    for grad in grads:
      self.assertTrue(grad is not None)

  def testWithIsRecomputeKwarg(self):

    kwarg_values = []

    @rev_block_lib.recompute_grad
    def layer_with_recompute(inputs, is_recomputing=False):
      kwarg_values.append(is_recomputing)
      out = core_layers.dense(inputs, 2)
      out = normalization_layers.batch_normalization(out, training=True)
      if is_recomputing:
        # Ensure that the updates are not duplicated by popping off the latest
        # 2 additions.
        update_ops = ops.get_collection_ref(ops.GraphKeys.UPDATE_OPS)
        update_ops.pop()
        update_ops.pop()
      return out

    x = array_ops.ones((2, 4), dtypes.float32)
    with variable_scope.variable_scope("layer1", use_resource=True):
      y = layer_with_recompute(x)
    loss = math_ops.reduce_sum(y)
    tvars = variables.trainable_variables()
    gradients_impl.gradients(loss, [x] + tvars)

    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    self.assertEqual(2, len(update_ops))
    self.assertEqual([False, True], kwarg_values)

  def testWithoutVariables(self):

    def concat_n(layer_list, num_inputs):
      return math_ops.reduce_sum(
          array_ops.concat([x for x in layer_list[-num_inputs:]], axis=-1),
          axis=1, keepdims=True)

    @rev_block_lib.recompute_grad
    def concat_n_wrap(*args):
      return concat_n(args, 3)

    # DenseNet-style layers
    layer_list = [random_ops.random_uniform((4, 8))]
    for _ in range(5):
      layer_list.append(math_ops.sqrt(concat_n_wrap(*layer_list)))

    grads = gradients_impl.gradients(layer_list[-1], layer_list[0])
    with self.test_session() as sess:
      sess.run(grads)

  def testErrorOnClosedOverTensor(self):
    x = random_ops.random_uniform((4, 8))
    y = random_ops.random_uniform((4, 8))
    z = x * y

    with self.assertRaisesWithPredicateMatch(ValueError, "closes over"):
      @rev_block_lib.recompute_grad
      def fn_with_capture(a):  # pylint: disable=unused-variable
        return a * z


if __name__ == "__main__":
  test.main()
