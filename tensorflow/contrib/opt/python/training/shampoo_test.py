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

"""Functional tests for AdaMoo optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.opt.python.training import shampoo
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

TOLERANCE = 1e-3


def np_power(mat_g, alpha):
  """Computes mat_g^alpha for a square symmetric matrix mat_g."""

  mat_u, diag_d, mat_v = np.linalg.svd(mat_g)
  diag_d = np.power(diag_d, alpha)
  return np.dot(np.dot(mat_u, np.diag(diag_d)), mat_v)


class ShampooTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Var', False), ('ResourceVar', True))
  def testBasicVector(self, use_resource_var):
    """Similar to the full Adagrad update."""

    size = 20
    init_var_np = np.zeros(size)
    grad_np = np.random.rand(size)
    grad_np_2 = np.random.rand(size)

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = constant_op.constant(grad_np, dtype=dtypes.float32)
      grad_2 = constant_op.constant(grad_np_2, dtype=dtypes.float32)

      opt = shampoo.ShampooOptimizer(global_step)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * mat_g^{-0.5} * grad
      # lr = 1
      mat_g = np.outer(grad_np, grad_np)
      mat_h = np_power(mat_g + 0.1 * np.eye(size), -0.5)
      new_val_np = init_var_np - np.dot(mat_h, grad_np)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g += np.outer(grad_np_2, grad_np_2)
      mat_h = np_power(mat_g + 0.1 * np.eye(size), -0.5)
      new_val_np -= np.dot(mat_h, grad_np_2)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(('Var', False), ('ResourceVar', True))
  def testBasicMatrix(self, use_resource_var):
    """Check update when gradient is a matrix."""
    size = [10, 5]
    init_var_np = np.zeros(size)
    grad_np = np.random.rand(size[0], size[1])
    grad_np_2 = np.random.rand(size[0], size[1])

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = constant_op.constant(grad_np, dtype=dtypes.float32)
      grad_2 = constant_op.constant(grad_np_2, dtype=dtypes.float32)

      opt = shampoo.ShampooOptimizer(global_step)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * mat_g1^{-0.25} * grad * mat_g2^{-0.25}
      # lr = 1
      mat_g1 = np.dot(grad_np, grad_np.transpose())
      mat_left = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.25)
      mat_g2 = np.dot(grad_np.transpose(), grad_np)
      mat_right = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.25)
      new_val_np = init_var_np - np.dot(np.dot(mat_left, grad_np), mat_right)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g1 += np.dot(grad_np_2, grad_np_2.transpose())
      mat_left = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.25)
      mat_g2 += np.dot(grad_np_2.transpose(), grad_np_2)
      mat_right = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.25)
      new_val_np -= np.dot(np.dot(mat_left, grad_np_2), mat_right)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  def _testBasicTensor(self, use_iterative_root, use_resource_var):
    """Check update when gradient is a tensor.

    Args:
      use_iterative_root: use iterative power method or SVD to find nth roots.
      use_resource_var: use resource var as variables.
    """
    size = [10, 5, 7]
    init_var_np = np.zeros(size)
    grad_np = np.random.rand(size[0], size[1], size[2])
    grad_np_2 = np.random.rand(size[0], size[1], size[2])

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = constant_op.constant(grad_np, dtype=dtypes.float32)
      grad_2 = constant_op.constant(grad_np_2, dtype=dtypes.float32)

      opt = shampoo.ShampooOptimizer(global_step,
                                     use_iterative_root=use_iterative_root)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * Prod_i mat_g_i^{-0.5/3} grad
      # lr = 1
      mat_g1 = np.tensordot(grad_np, grad_np, axes=([1, 2], [1, 2]))
      mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
      mat_g2 = np.tensordot(grad_np, grad_np, axes=([0, 2], [0, 2]))
      mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
      mat_g3 = np.tensordot(grad_np, grad_np, axes=([0, 1], [0, 1]))
      mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

      precond_grad = np.tensordot(grad_np, mat_g1_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
      new_val_np = init_var_np - precond_grad

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g1 += np.tensordot(grad_np_2, grad_np_2, axes=([1, 2], [1, 2]))
      mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
      mat_g2 += np.tensordot(grad_np_2, grad_np_2, axes=([0, 2], [0, 2]))
      mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
      mat_g3 += np.tensordot(grad_np_2, grad_np_2, axes=([0, 1], [0, 1]))
      mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

      precond_grad = np.tensordot(grad_np_2, mat_g1_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
      new_val_np -= precond_grad

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(
      ('SVDWithVar', False, False),
      ('SVDWithResourceVar', False, True),
      ('IterRootWithVar', True, False),
      ('IterRootWithResourceVar', True, True),
  )
  def testBasicTensor(self, use_iterative_root, use_resource_var):
    self._testBasicTensor(use_iterative_root, use_resource_var)

  @parameterized.named_parameters(('Var', False), ('ResourceVar', True))
  def testLargeVector(self, use_resource_var):
    """This is just the diagonal Adagrad update."""

    size = 2000
    init_var_np = np.zeros(size)
    grad_np = np.random.rand(size)
    grad_np_2 = np.random.rand(size)

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = constant_op.constant(grad_np, dtype=dtypes.float32)
      grad_2 = constant_op.constant(grad_np_2, dtype=dtypes.float32)

      opt = shampoo.ShampooOptimizer(global_step)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * gg^{-0.5} * grad
      # lr = 1
      mat_g = grad_np * grad_np + 0.1
      new_val_np = init_var_np - np.power(mat_g, -0.5) * grad_np

      self.assertAllCloseAccordingToType(new_val_np, new_val)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g += grad_np_2 * grad_np_2
      new_val_np -= np.power(mat_g, -0.5) * grad_np_2

      self.assertAllCloseAccordingToType(new_val_np, new_val)

  @parameterized.named_parameters(('Var', False), ('ResourceVar', True))
  def testLargeMatrix(self, use_resource_var):
    """Gradient is a matrix, one of whose dimensions is large.

    We do diagonal updates for large dimensions.

    Args:
      use_resource_var: use resource var as variables.
    """

    size = [2000, 3]
    init_var_np = np.zeros(size)
    grad_np = np.random.rand(size[0], size[1])
    grad_np_2 = np.random.rand(size[0], size[1])

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = constant_op.constant(grad_np, dtype=dtypes.float32)
      grad_2 = constant_op.constant(grad_np_2, dtype=dtypes.float32)

      opt = shampoo.ShampooOptimizer(global_step)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * mat_left * grad * mat_right
      # where the mat_left * grad is just element-wise product,
      # with broadcasting
      # lr = 1

      mat_g1 = np.sum(grad_np * grad_np, axis=1, keepdims=True)
      mat_left = np.power(mat_g1 + 0.1, -0.25)
      mat_g2 = np.dot(grad_np.transpose(), grad_np)
      mat_right = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.25)
      new_val_np = init_var_np - np.dot(grad_np * mat_left, mat_right)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g1 += np.sum(grad_np_2 * grad_np_2, axis=1, keepdims=True)
      mat_left = np.power(mat_g1 + 0.1, -0.25)
      mat_g2 += np.dot(grad_np_2.transpose(), grad_np_2)
      mat_right = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.25)
      new_val_np -= np.dot(grad_np_2 * mat_left, mat_right)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(('Var', False))
  def testSparseUpdateLarge(self, use_resource_var):
    """Check update when gradient is of type IndexSlices.

    We do diagonal updates for the first dimension, unless it is very small.

    Args:
      use_resource_var: use resource var as variables.
    """
    size = [2000, 3]
    sample_size_1 = 100
    init_var_np = np.zeros(size)
    grad_indices = np.sort(np.random.choice(np.arange(size[0]), sample_size_1,
                                            replace=False))
    grad_np = np.random.rand(sample_size_1, size[1])

    sample_size_2 = 7
    grad_indices_2 = np.sort(np.random.choice(np.arange(size[0]), sample_size_2,
                                              replace=False))
    grad_np_2 = np.random.rand(sample_size_2, size[1])

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = ops.IndexedSlices(
          constant_op.constant(grad_np, dtype=dtypes.float32),
          constant_op.constant(grad_indices),
          constant_op.constant(size))
      grad_2 = ops.IndexedSlices(
          constant_op.constant(grad_np_2, dtype=dtypes.float32),
          constant_op.constant(grad_indices_2),
          constant_op.constant(size))

      opt = shampoo.ShampooOptimizer(global_step)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * mat_left * grad * mat_right
      # where the mat_left * grad is just element-wise product,
      # with broadcasting
      # lr = 1
      # In this case the update lr * mat_left * grad * mat_right is
      # of size 10 x 2.
      # So the correct indices of var need to be updated.

      mat_g1 = np.sum(grad_np * grad_np, axis=1, keepdims=True)
      mat_g1_acc = np.zeros((size[0], 1))
      mat_g1_acc[grad_indices] += mat_g1
      mat_left = np.power(mat_g1 + 0.1, -0.25)
      mat_g2 = np.dot(grad_np.transpose(), grad_np)
      mat_right = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.25)
      new_val_np = init_var_np
      new_val_np[grad_indices, :] -= np.dot(grad_np * mat_left, mat_right)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g1 = np.sum(grad_np_2 * grad_np_2, axis=1, keepdims=True)
      mat_g1_acc[grad_indices_2] += mat_g1
      mat_left = np.power(mat_g1_acc[grad_indices_2] + 0.1, -0.25)
      mat_g2 += np.dot(grad_np_2.transpose(), grad_np_2)
      mat_right = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.25)
      new_val_np[grad_indices_2, :] -= np.dot(grad_np_2 * mat_left, mat_right)

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  def _testSparseUpdateSmall(self, use_iterative_root, use_resource_var):
    """Gradient is of type IndexSlices, but the first dimension is small.

    We create dense gradient and do the full update with SVD etc.

    Args:
      use_iterative_root: use iterative power method or SVD to find nth roots.
      use_resource_var: use resource var as variables.
    """

    size = [100, 3, 5]
    sample_size = 10
    init_var_np = np.zeros(size)
    grad_indices = np.sort(np.random.choice(np.arange(size[0]), sample_size,
                                            replace=False))
    grad_np = np.random.rand(sample_size, size[1], size[2])

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = ops.IndexedSlices(
          constant_op.constant(grad_np, dtype=dtypes.float32),
          constant_op.constant(grad_indices),
          constant_op.constant(size))

      opt = shampoo.ShampooOptimizer(global_step,
                                     use_iterative_root=use_iterative_root)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * Prod_i mat_g_i^{-0.125} grad
      # lr = 1
      grad_dense = np.zeros_like(init_var_np)
      grad_dense[grad_indices] = grad_np

      mat_g1 = np.tensordot(grad_dense, grad_dense, axes=([1, 2], [1, 2]))
      mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
      mat_g2 = np.tensordot(grad_dense, grad_dense, axes=([0, 2], [0, 2]))
      mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
      mat_g3 = np.tensordot(grad_dense, grad_dense, axes=([0, 1], [0, 1]))
      mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

      precond_grad = np.tensordot(grad_dense, mat_g1_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
      new_val_np = init_var_np - precond_grad

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(
      ('SVDWithVar', False, False),
      ('SVDWithResourceVar', False, True),
      ('IterRootWithVar', True, False),
      ('IterRootWithResourceVar', True, True),
  )
  def testSparseUpdateSmall(self, use_iterative_root, use_resource_var):
    self._testSparseUpdateSmall(use_iterative_root, use_resource_var)

  def _testBasicTensorWithMomentum(self, use_iterative_root, use_resource_var):
    """Check update with momentum when gradient is a tensor.

    Args:
      use_iterative_root: use iterative power method or SVD to find nth roots.
      use_resource_var: use resource var as variables.
    """
    size = [10, 5, 7]
    init_var_np = np.zeros(size)
    grad_np = np.random.rand(size[0], size[1], size[2])
    grad_np_2 = np.random.rand(size[0], size[1], size[2])
    gbar_decay = 0.9
    gbar_weight = 0.1

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = constant_op.constant(grad_np, dtype=dtypes.float32)
      grad_2 = constant_op.constant(grad_np_2, dtype=dtypes.float32)

      opt = shampoo.ShampooOptimizer(global_step, gbar_decay=gbar_decay,
                                     gbar_weight=gbar_weight,
                                     use_iterative_root=use_iterative_root)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      update_2 = opt.apply_gradients(zip([grad_2], [var]),
                                     global_step=global_step)
      variables.global_variables_initializer().run()

      # Run a step of Shampoo
      update.run()
      new_val = sess.run(var)

      # let up compute this in numpy
      # Update rule is var = var - lr * Prod_i mat_g_i^{-0.5/3} grad
      # lr = 1
      mat_g1 = np.tensordot(grad_np, grad_np, axes=([1, 2], [1, 2]))
      mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
      mat_g2 = np.tensordot(grad_np, grad_np, axes=([0, 2], [0, 2]))
      mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
      mat_g3 = np.tensordot(grad_np, grad_np, axes=([0, 1], [0, 1]))
      mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

      gbar_np = gbar_weight * grad_np
      precond_grad = np.tensordot(gbar_np, mat_g1_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
      new_val_np = init_var_np - precond_grad

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

      # Run another step of Shampoo
      update_2.run()
      new_val = sess.run(var)

      mat_g1 += np.tensordot(grad_np_2, grad_np_2, axes=([1, 2], [1, 2]))
      mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
      mat_g2 += np.tensordot(grad_np_2, grad_np_2, axes=([0, 2], [0, 2]))
      mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
      mat_g3 += np.tensordot(grad_np_2, grad_np_2, axes=([0, 1], [0, 1]))
      mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

      gbar_np_2 = gbar_decay * gbar_np + gbar_weight * grad_np_2
      precond_grad = np.tensordot(gbar_np_2, mat_g1_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
      precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
      new_val_np -= precond_grad

      self.assertAllCloseAccordingToType(new_val_np, new_val,
                                         atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(
      ('SVDWithVar', False, False),
      ('SVDWithResourceVar', False, True),
      ('IterRootWithVar', True, False),
      ('IterRootWithResourceVar', True, True),
  )
  def testBasicTensorWithMomentum(self, use_iterative_root, use_resource_var):
    self._testBasicTensorWithMomentum(use_iterative_root, use_resource_var)

  def _testDelayedSVD(self, use_iterative_root, use_resource_var):
    """Performing the SVD every nth step.

    Args:
      use_iterative_root: use iterative power method or SVD to find nth roots.
      use_resource_var: use resource var as variables.
    """
    size = [10, 5, 7]
    init_var_np = np.zeros(size).astype(np.float32)
    iterations = 20
    svd_interval = 5
    grad_np = np.random.rand(
        iterations, size[0], size[1], size[2]).astype(np.float32)
    mat_g1_a = np.eye(size[0])
    mat_g1 = np.zeros_like(mat_g1_a)
    mat_g2_a = np.eye(size[1])
    mat_g2 = np.zeros_like(mat_g2_a)
    mat_g3_a = np.eye(size[2])
    mat_g3 = np.zeros_like(mat_g3_a)

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = array_ops.placeholder(dtypes.float32, shape=size)

      opt = shampoo.ShampooOptimizer(global_step, svd_interval=svd_interval,
                                     use_iterative_root=use_iterative_root)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)
      new_val_np = init_var_np

      # Run n steps of Shampoo
      for i in range(iterations):
        _ = sess.run(update, feed_dict={grad: grad_np[i]})
        new_val = sess.run(var)

        # let up compute this in numpy
        # Update rule is var = var - lr * Prod_i mat_g_i^{-0.5/3} grad
        # lr = 1
        mat_g1 += np.tensordot(grad_np[i], grad_np[i], axes=([1, 2], [1, 2]))
        mat_g2 += np.tensordot(grad_np[i], grad_np[i], axes=([0, 2], [0, 2]))
        mat_g3 += np.tensordot(grad_np[i], grad_np[i], axes=([0, 1], [0, 1]))
        if (i + 1) % svd_interval == 0:
          mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
          mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
          mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

        precond_grad = np.tensordot(grad_np[i], mat_g1_a, axes=([0], [0]))
        precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
        precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
        new_val_np -= precond_grad

        self.assertAllCloseAccordingToType(new_val_np, new_val,
                                           atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(
      ('SVDWithVar', False, False),
      ('SVDWithResourceVar', False, True),
      ('IterRootWithVar', True, False),
      ('IterRootWithResourceVar', True, True),
  )
  def testDelayedSVD(self, use_iterative_root, use_resource_var):
    self._testDelayedSVD(use_iterative_root, use_resource_var)

  def _testDelayedPrecondUpdate(self, use_iterative_root, use_resource_var):
    """Update the squared sum every nth step, drop the other steps.

    Args:
      use_iterative_root: use iterative power method or SVD to find nth roots.
      use_resource_var: use resource var as variables.
    """
    size = [10, 5, 7]
    init_var_np = np.zeros(size).astype(np.float32)
    iterations = 100
    grad_np = np.random.rand(
        iterations, size[0], size[1], size[2]).astype(np.float32)
    svd_interval = 20
    precond_update_interval = 5
    mat_g1_a = np.eye(size[0])
    mat_g1 = np.zeros_like(mat_g1_a)
    mat_g2_a = np.eye(size[1])
    mat_g2 = np.zeros_like(mat_g2_a)
    mat_g3_a = np.eye(size[2])
    mat_g3 = np.zeros_like(mat_g3_a)

    with self.test_session() as sess:
      global_step = variables.Variable(
          0, dtype=dtypes.int64, use_resource=use_resource_var)
      var = variables.Variable(
          init_var_np, dtype=dtypes.float32, use_resource=use_resource_var)
      grad = array_ops.placeholder(dtypes.float32, shape=size)

      opt = shampoo.ShampooOptimizer(
          global_step, svd_interval=svd_interval,
          precond_update_interval=precond_update_interval,
          use_iterative_root=use_iterative_root)
      update = opt.apply_gradients(zip([grad], [var]),
                                   global_step=global_step)
      variables.global_variables_initializer().run()

      init_val = sess.run(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)
      new_val_np = init_var_np

      # Run n steps of Shampoo
      for i in range(iterations):
        _ = sess.run(update, feed_dict={grad: grad_np[i]})
        new_val = sess.run(var)

        # let up compute this in numpy
        # Update rule is var = var - lr * Prod_i mat_g_i^{-0.5/3} grad
        # lr = 1
        if (i + 1) % precond_update_interval == 0:
          mat_g1 += (np.tensordot(grad_np[i], grad_np[i], axes=([1, 2], [1, 2]))
                     * precond_update_interval)
          mat_g2 += (np.tensordot(grad_np[i], grad_np[i], axes=([0, 2], [0, 2]))
                     * precond_update_interval)
          mat_g3 += (np.tensordot(grad_np[i], grad_np[i], axes=([0, 1], [0, 1]))
                     * precond_update_interval)

        if (i + 1) % svd_interval == 0:
          mat_g1_a = np_power(mat_g1 + 0.1 * np.eye(size[0]), -0.5/3.0)
          mat_g2_a = np_power(mat_g2 + 0.1 * np.eye(size[1]), -0.5/3.0)
          mat_g3_a = np_power(mat_g3 + 0.1 * np.eye(size[2]), -0.5/3.0)

        precond_grad = np.tensordot(grad_np[i], mat_g1_a, axes=([0], [0]))
        precond_grad = np.tensordot(precond_grad, mat_g2_a, axes=([0], [0]))
        precond_grad = np.tensordot(precond_grad, mat_g3_a, axes=([0], [0]))
        new_val_np -= precond_grad

        self.assertAllCloseAccordingToType(new_val_np, new_val,
                                           atol=TOLERANCE, rtol=TOLERANCE)

  @parameterized.named_parameters(
      ('SVDWithVar', False, False),
      ('SVDWithResourceVar', False, True),
      ('IterRootWithVar', True, False),
      ('IterRootWithResourceVar', True, True),
  )
  def testDelayedPrecondUpdate(self, use_iterative_root, use_resource_var):
    self._testDelayedPrecondUpdate(use_iterative_root, use_resource_var)


if __name__ == '__main__':
  test.main()
