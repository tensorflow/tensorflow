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

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework.experimental import _unified_api
from tensorflow.python.framework.experimental import context_stack as context_lib
from tensorflow.python.framework.experimental import def_function
from tensorflow.python.framework.experimental import math_ops
from tensorflow.python.framework.experimental import tape as tape_lib
from tensorflow.python.platform import test

SetTracingImplementation = _unified_api.SetTracingImplementation
TensorCastHelper = _unified_api.EagerTensorToImmediateExecutionTensorHandle


def get_immediate_execution_context():
  context.context().ensure_initialized()
  return _unified_api.EagerContextToImmediateExecutionContext(
      context.context()._handle)


class UnifiedApiTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testAdd(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      return math_ops.add(a, b)

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
        result = math_ops.add(a, b)
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
  def testTimeMNISTGrad(self, use_mlir):
    if use_mlir:
      SetTracingImplementation("mlir")
 
    # 2-layer Net Model
    def model(X, W1, W2, labels):
      with tape_lib.GradientTape() as tape:
        tape.watch(W1)
        tape.watch(W2)
        mm = math_ops.matmul(X, W1)
        hidden = math_ops.relu(mm)
        scores = math_ops.matmul(hidden, W2)
        loss = math_ops.softmax_loss(scores, labels)
      
      grads = tape.gradient(loss, [W1, W2])
      return grads
 
    times_recorded = []
    num_iters = 500
    for i in range(num_iters):
      t1 = time.time()
      with context_lib.set_default(get_immediate_execution_context()):
        X = eager_ctx.CreateFloatTensorHandle([1., 2., 3., 4.], [2,2])
        W1 = eager_ctx.CreateFloatTensorHandle([-1., 10., 0.5, 1.], [2,2])
        W2 = eager_ctx.CreateFloatTensorHandle([.1, .2, .3, -.5], [2,2])
        labels = eager_ctx.CreateIntTensorHandle([1,1], [2])
      
        eager_outputs = model(X, W1, W2, labels)
        t2 = time.time()
  
        delta_t = t2 - t1
        times_recorded.append(delta_t)

    print(times_recorded) # Need this for comparison to default grads using TF

if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
