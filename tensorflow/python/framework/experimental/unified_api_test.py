# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Unified APIs' python bindings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.experimental import _unified_api
from tensorflow.python.framework.experimental import context_stack as context_lib
from tensorflow.python.framework.experimental import def_function
from tensorflow.python.framework.experimental import math_ops as unified_math_ops
from tensorflow.python.framework.experimental import nn_ops as unified_nn_ops
from tensorflow.python.framework.experimental import tape as tape_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

SetTracingImplementation = _unified_api.SetTracingImplementation
TensorCastHelper = _unified_api.EagerTensorToImmediateExecutionTensorHandle


def get_immediate_execution_context():
  context._reset_context()
  context.context().ensure_initialized()
  return _unified_api.EagerContextToImmediateExecutionContext(
      context.context()._handle)


def maybe_cast(t, perform_cast):
  if perform_cast:
    return TensorCastHelper(t)
  return t


class UnifiedApiTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testAdd(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      return unified_math_ops.add(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_output = def_function.function(model)(a, b)
      self.assertAllEqual(func_output.numpy(), [4., 6.])

      eager_output = model(a, b)
      self.assertAllEqual(eager_output.numpy(), [4., 6.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testAddGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.add(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertAllEqual(func_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(func_outputs[1].numpy(), [1.0, 1.0])

      eager_outputs = model(a, b)
      self.assertAllEqual(eager_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(eager_outputs[1].numpy(), [1.0, 1.0])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testRelu(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(t):
      return unified_nn_ops.relu(t)

    with context_lib.set_default(get_immediate_execution_context()):
      positive = TensorCastHelper(constant_op.constant([1.]))
      negative = TensorCastHelper(constant_op.constant([-1.]))

      model_fn = def_function.function(model)
      func_output = model_fn(positive)
      self.assertAllEqual(func_output.numpy(), [1.])
      func_output = model_fn(negative)
      self.assertAllEqual(func_output.numpy(), [0.])

      eager_output = model(positive)
      self.assertAllEqual(eager_output.numpy(), [1.])
      eager_output = model(negative)
      self.assertAllEqual(eager_output.numpy(), [0.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testReluGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(t):
      with tape_lib.GradientTape() as tape:
        tape.watch(t)
        result = unified_nn_ops.relu(t)
      grads = tape.gradient(result, t)
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      positive = TensorCastHelper(constant_op.constant([1.]))
      negative = TensorCastHelper(constant_op.constant([-1.]))

      model_fn = def_function.function(model)
      func_output = model_fn(positive)
      self.assertAllEqual(func_output.numpy(), [1.])
      func_output = model_fn(negative)
      self.assertAllEqual(func_output.numpy(), [0.])

      eager_output = model(positive)
      self.assertAllEqual(eager_output.numpy(), [1.])
      eager_output = model(negative)
      self.assertAllEqual(eager_output.numpy(), [0.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testNeg(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      return unified_math_ops.neg(a)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))

      func_output = def_function.function(model)(a)
      self.assertAllEqual(func_output.numpy(), [-2.])

      eager_output = model(a)
      self.assertAllEqual(eager_output.numpy(), [-2.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testNegGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        result = unified_math_ops.neg(a)
      grads = tape.gradient(result, a)
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))

      func_outputs = def_function.function(model)(a)
      self.assertAllEqual(func_outputs.numpy(), [-1.0])

      eager_outputs = model(a)
      self.assertAllEqual(eager_outputs.numpy(), [-1.0])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testSub(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      return unified_math_ops.sub(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_output = def_function.function(model)(a, b)
      self.assertAllEqual(func_output.numpy(), [-2., -2.])

      eager_output = model(a, b)
      self.assertAllEqual(eager_output.numpy(), [-2., -2.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testSubGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.sub(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertAllEqual(func_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(func_outputs[1].numpy(), [-1.0, -1.0])

      eager_outputs = model(a, b)
      self.assertAllEqual(eager_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(eager_outputs[1].numpy(), [-1.0, -1.0])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testMul(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      return unified_math_ops.mul(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_output = def_function.function(model)(a, b)
      self.assertAllEqual(func_output.numpy(), [3., 8.])

      eager_output = model(a, b)
      self.assertAllEqual(eager_output.numpy(), [3., 8.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testMulGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.mul(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertAllEqual(func_outputs[0].numpy(), [3., 4.])
      self.assertAllEqual(func_outputs[1].numpy(), [1., 2.])

      eager_outputs = model(a, b)
      self.assertAllEqual(eager_outputs[0].numpy(), [3., 4.])
      self.assertAllEqual(eager_outputs[1].numpy(), [1., 2.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testLog1p(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      return unified_math_ops.log1p(a)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1.]))

      func_output = def_function.function(model)(a)
      self.assertArrayNear(func_output.numpy(), [0.69314], 0.001)

      eager_output = model(a)
      self.assertArrayNear(eager_output.numpy(), [0.69314], 0.001)

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testLog1pGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        result = unified_math_ops.log1p(a)
      grads = tape.gradient(result, a)
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1.]))

      func_outputs = def_function.function(model)(a)
      self.assertArrayNear(func_outputs.numpy(), [0.5], 0.001)

      eager_outputs = model(a)
      self.assertArrayNear(eager_outputs.numpy(), [0.5], 0.001)

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testDivNoNan(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      return unified_math_ops.div_no_nan(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))
      b = TensorCastHelper(constant_op.constant([4.]))

      func_output = def_function.function(model)(a, b)
      self.assertArrayNear(func_output.numpy(), [0.5], 0.001)

      eager_output = model(a, b)
      self.assertArrayNear(eager_output.numpy(), [0.5], 0.001)

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testDivNoNanGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.div_no_nan(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))
      b = TensorCastHelper(constant_op.constant([4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertArrayNear(func_outputs[0].numpy(), [0.25], 0.001)
      self.assertArrayNear(func_outputs[1].numpy(), [-0.125], 0.001)

      eager_outputs = model(a, b)
      self.assertArrayNear(eager_outputs[0].numpy(), [0.25], 0.001)
      self.assertArrayNear(eager_outputs[1].numpy(), [-0.125], 0.001)


class UnifiedTapeBenchmark(test.Benchmark):

  def _computeMnistMlpGrads(self, math_ops_lib, nn_ops_lib, backprop_lib, cast,
                            num_iters, hidden_layers, hidden_size, batch_size):
    batch_size = 1
    image_size = 28 * 28
    num_classes = 10

    def model(x, hidden_weights, softmax_weight, labels):
      with backprop_lib.GradientTape() as tape:
        for weight in hidden_weights + [softmax_weight]:
          tape.watch(weight)
        for hidden_weight in hidden_weights:
          x = math_ops_lib.mat_mul(x, hidden_weight)
          x = nn_ops_lib.relu(x)
        logits = math_ops_lib.mat_mul(x, softmax_weight)
        loss = nn_ops_lib.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

      grads = tape.gradient(loss, hidden_weights + [softmax_weight])
      return grads

    x = maybe_cast(array_ops.ones([batch_size, image_size]), cast)
    hidden_weights = []
    for i in range(hidden_layers):
      hidden_weights.append(
          maybe_cast(
              random_ops.random_uniform(
                  [hidden_size if i else image_size, hidden_size]), cast))
    softmax_weight = maybe_cast(
        random_ops.random_uniform([hidden_size, num_classes]), cast)
    labels = maybe_cast(array_ops.zeros([batch_size], dtype=dtypes.int32), cast)

    with context_lib.set_default(get_immediate_execution_context()):
      # Warm up.
      for _ in range(10):
        model(x, hidden_weights, softmax_weight, labels)
      runtimes = timeit.repeat(
          lambda: model(x, hidden_weights, softmax_weight, labels),
          repeat=num_iters,
          number=10)
    return min(runtimes) / 10

  def benchmarkTwoHiddenLayerMnistEagerUnified(self):
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        unified_math_ops,
        unified_nn_ops,
        tape_lib,
        True,
        num_iters,
        hidden_layers=2,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TwoHiddenLayerMnistEagerUnified",
        iters=num_iters,
        wall_time=duration)

  def benchmarkTwoHiddenLayerMnistEager(self):
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        math_ops,
        nn_ops,
        backprop,
        False,
        num_iters,
        hidden_layers=2,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TwoHiddenLayerMnistEager", iters=num_iters, wall_time=duration)

  def benchmarkTenHiddenLayerMnistEagerUnified(self):
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        unified_math_ops,
        unified_nn_ops,
        tape_lib,
        True,
        num_iters,
        hidden_layers=10,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TenHiddenLayerMnistEagerUnified",
        iters=num_iters,
        wall_time=duration)

  def benchmarkTenHiddenLayerMnistEager(self):
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        math_ops,
        nn_ops,
        backprop,
        False,
        num_iters,
        hidden_layers=10,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TenHiddenLayerMnistEager", iters=num_iters, wall_time=duration)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
