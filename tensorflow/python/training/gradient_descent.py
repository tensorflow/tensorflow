"""GradientDescent for TensorFlow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
# pylint: disable=unused-import
from tensorflow.python.ops import math_ops
# pylint: enable=unused-import
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class GradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.

  @@__init__
  """

  def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
    """Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operation.s
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(GradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate

  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        self._learning_rate_tensor,
        grad,
        use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):
    delta = ops.IndexedSlices(grad.values * self._learning_rate_tensor,
                              grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
