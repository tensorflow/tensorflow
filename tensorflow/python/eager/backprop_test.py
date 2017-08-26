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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.eager import tensor
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import training


class BackpropTest(test.TestCase):

  def testAggregateGradients(self):

    def fn(x):
      ind1 = tensor.Tensor(np.array([0, 1]))
      ind2 = tensor.Tensor(np.array([2, 3]))
      ind3 = tensor.Tensor(np.array([1, 3]))
      # A mixture of IndexedSlices and dense tensor to aggregate.
      g1 = embedding_ops.embedding_lookup(x, ind1)
      g2 = embedding_ops.embedding_lookup(x, ind2)
      g3 = embedding_ops.embedding_lookup(x, ind3)
      g4 = math_ops.reduce_sum(x * tensor.Tensor(2.0))
      return g1 * g2 * g3 * g4

    var_np = np.random.rand(4, 2).astype(np.float32)
    var = tensor.Tensor(var_np)
    grad = backprop.gradients_function(fn, [0])(var)[0]

    with context.graph_mode(), self.test_session():
      tf_var = array_ops.constant(var_np, dtypes.float32)
      tf_ind1 = array_ops.constant([0, 1])
      tf_ind2 = array_ops.constant([2, 3])
      tf_ind3 = array_ops.constant([1, 3])
      tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
      tf_g2 = embedding_ops.embedding_lookup(tf_var, tf_ind2)
      tf_g3 = embedding_ops.embedding_lookup(tf_var, tf_ind3)
      tf_g4 = math_ops.reduce_sum(tf_var * 2.0, reduction_indices=(0, 1))
      tf_y = tf_g1 * tf_g2 * tf_g3 * tf_g4
      tf_grad = gradients.gradients(tf_y, [tf_var])[0]

      tf_dense_grad = math_ops.unsorted_segment_sum(
          tf_grad.values, tf_grad.indices, tf_grad.dense_shape[0])

      self.assertAllClose(grad.numpy(), tf_dense_grad.eval())

  def testImplicitGradWithResourceVariable(self):
    x = resource_variable_ops.ResourceVariable(
        initial_value=tensor.Tensor(1.0), name='x')

    def fn():
      tape.watch(x.handle)
      b = tensor.Tensor(2.0)
      c = math_ops.add(x.value(), b)
      return math_ops.add(c, tensor.Tensor(3.0))

    grad = backprop.implicit_grad(fn)()[0][1]
    self.assertEqual(grad.numpy(), 1.0)

  def testImplicitGradOverEmbeddingLookup(self):
    batch_size = 8
    embedding_size = 512
    vocab_size = 1000
    lrn_rate = 0.1
    random_init = random_ops.random_uniform([vocab_size, embedding_size])

    x = array_ops.ones((batch_size), dtypes.int64)
    embedding = resource_variable_ops.ResourceVariable(
        initial_value=random_init, dtype=dtypes.float32, name='embedding')

    def f():
      tape.watch(embedding.handle)
      embedded_x = embedding_ops.embedding_lookup(embedding, x)
      return tensor.Tensor(1.0, dtypes.float32) - embedded_x

    grad = backprop.implicit_grad(f)()[0][1]
    opt = training.GradientDescentOptimizer(lrn_rate)

    with context.graph_mode(), self.test_session():
      tf_x = array_ops.ones((batch_size), dtypes.int64)
      # TODO(ashankar,apassos): Change to ResourceVariable.
      tf_embedding = variables.Variable(
          random_init.numpy(), name='tf_embedding')
      tf_embedded_x = embedding_ops.embedding_lookup(tf_embedding, tf_x)
      tf_y = 1.0 - tf_embedded_x
      tf_grad = gradients.gradients(tf_y, [tf_embedding])[0]
      tf_opt = training.GradientDescentOptimizer(0.1)
      tf_embedding.initializer.run()

      self.assertAllClose(tf_grad.indices.eval(), grad.indices.numpy())
      self.assertAllClose(tf_grad.values.eval(), grad.values.numpy())

      tf_opt.apply_gradients([(tf_grad, tf_embedding)]).run()
      expected = tf_embedding.eval()
    opt.apply_gradients([(grad, embedding)])
    self.assertAllClose(expected, embedding.read_value().numpy())

  def testGradientNone(self):

    def loss(x, l):
      return math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits(logits=x, labels=l),
          tensor.Tensor([0]))

    logits = tensor.Tensor([[0.0, 0.0]])
    labels = tensor.Tensor([[1.0, 0.0]])
    # softmax_cross_entropy_with_logits returns two outputs and in this case the
    # gradient wrt the second is None.
    g, = backprop.gradients_function(loss, [0])(logits, labels)
    self.assertAllEqual(g.numpy(), [[-0.5, 0.5]])

  def testSecondGrad(self):

    def first(x):
      l = tensor.Tensor([[0.0]])
      x = nn_ops.softmax_cross_entropy_with_logits(labels=l, logits=x)
      x = math_ops.reduce_sum(x, tensor.Tensor([0]))
      return x

    def second(x):
      grad = backprop.gradients_function(first, [0])(x)[0]
      return math_ops.reduce_sum(grad, tensor.Tensor([0]))

    f = tensor.Tensor([[0.1]])
    grad = backprop.gradients_function(second, [0])(f)[0]
    self.assertAllEqual([[0.0]], grad.numpy())

  def testGPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    def fn(x):
      with context.device('/gpu:0'):
        b = tensor.Tensor(2.0)
        c = math_ops.add(x.as_gpu_tensor(), b)
        # TODO(apassos): remove as_cpu_tensor below by making TensorVSPace aware
        # of devices.
        return math_ops.add(c, tensor.Tensor(3.0)).as_cpu_tensor()

    grad = backprop.gradients_function(fn, [0])(tensor.Tensor(1.0))[0]
    self.assertEqual(grad.numpy(), 1.0)

  def testGPUImplicitGrad(self):
    if not context.context().num_gpus():
      self.skipTest('No GPU found')
    with context.device('gpu:0'):
      v = resource_variable_ops.ResourceVariable(tensor.Tensor(1.0), name='v')

    def f():
      with context.device('gpu:0'):
        tape.watch(v.handle)
        return v.read_value()

    self.assertEqual(
        backprop.implicit_grad(f)()[0][1].as_cpu_tensor().numpy(), 1.0)

  def testCPU(self):

    def fn(x):
      b = tensor.Tensor(2.0)
      c = math_ops.add(x, b)
      return math_ops.add(c, tensor.Tensor(3.0))

    grad = backprop.gradients_function(fn, [0])(tensor.Tensor(1.0))[0]
    self.assertEqual(grad.numpy(), 1.0)

  def testTensorCopyGPU2CPU2GPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    def f(a, b):
      return a.as_cpu_tensor() + b.as_cpu_tensor()

    with context.device('/gpu:0'):
      a = tensor.Tensor(1.0)
      b = tensor.Tensor(2.0)

    grad = backprop.gradients_function(f, [0])(a, b)[0]
    self.assertEqual(grad.numpy(), 1.0)

  def testEmptyParams(self):

    def fn(a, b):
      return a * b

    x = tensor.Tensor(1.0)
    y = tensor.Tensor(2.0)
    dx, dy = backprop.gradients_function(fn)(x, y)
    self.assertAllEqual(dx.numpy(), y.numpy())
    self.assertAllEqual(dy.numpy(), x.numpy())

  def testTensorCopyCPU2GPU2CPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # forward: a (cpu->gpu) -> add (gpu) -> c (gpu->cpu) -> add (cpu) -> e (cpu)
    # back: e (cpu) -> add (cpu) -> c (cpu->gpu) -> add (gpu) -> grad (gpu->cpu)
    def f(a, b):
      with context.device('/gpu:0'):
        c = math_ops.add(a.as_gpu_tensor(0), b.as_gpu_tensor(0))
      return math_ops.add(c.as_cpu_tensor(), tensor.Tensor(3.0))

    with context.device('/cpu:0'):
      a = tensor.Tensor(1.0)
      b = tensor.Tensor(2.0)

    grad = backprop.gradients_function(f, [0])(a, b)[0]
    self.assertEqual(grad.numpy(), 1.0)

  def testGetAttrType(self):
    typ = backprop.op_attr_type('Add', 'T')
    self.assertEqual(typ, pywrap_tensorflow.TF_ATTR_TYPE)

  def testGetAttrList(self):
    typ = backprop.op_attr_type('MaxPool', 'ksize')
    self.assertEqual(typ, [pywrap_tensorflow.TF_ATTR_INT])

  def testMakeAttrType(self):
    self.assertEqual(dtypes.float32,
                     backprop.make_attr(pywrap_tensorflow.TF_ATTR_TYPE, 1))

  def testMakeAttrTypeList(self):
    self.assertEqual([dtypes.float32],
                     backprop.make_attr([pywrap_tensorflow.TF_ATTR_TYPE], [1]))

  def testMakeAttrShape(self):
    for s in ([], None, [1, 2, 3], [None, None], [1, None, 3]):
      expected = tensor_shape.TensorShape(s).as_proto()
      actual = backprop.make_attr(pywrap_tensorflow.TF_ATTR_SHAPE, s)
      self.assertEqual(
          expected,
          actual,
          msg=('For shape %r, expected %r != %r actual' % (s, expected,
                                                           actual)))

  def testMakeAttrShapeList(self):
    shape_list = [[], None, [1, 2, 3], [None, None], [1, None, 3]]
    self.assertEqual(
        [tensor_shape.TensorShape(s).as_proto() for s in shape_list],
        backprop.make_attr([pywrap_tensorflow.TF_ATTR_SHAPE], shape_list))


if __name__ == '__main__':
  test.main()
