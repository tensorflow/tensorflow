"""Adagrad for TensorFlow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class AdagradOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adagrad algorithm.

  @@__init__
  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1,
               use_locking=False, name="Adagrad"):
    """Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the initial_accumulator_value is invalid.
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(AdagradOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._initial_accumulator_value = initial_accumulator_value
    # Created in Initialize.
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      val = constant_op.constant(self._initial_accumulator_value,
                                 shape=v.get_shape())
      self._get_or_make_slot(v, val, "accumulator", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")

  def _apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return training_ops.apply_adagrad(
        var, acc, self._learning_rate_tensor, grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return training_ops.sparse_apply_adagrad(
        var, acc, self._learning_rate_tensor, grad.values, grad.indices,
        use_locking=self._use_locking)
