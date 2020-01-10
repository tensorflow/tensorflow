"""FTRL-Proximal for Tensor Flow."""
from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


def _Solve(a, b, c):
  """Return solution of a quadratic minimization.

  The optimization equation is:
       f(a, b, c) = argmin_w{1/2 * a * w^2 + b * w + c * |w|}
  we get optimal solution w*:
       w* = -(b - sign(b)*c)/a if |b| > c else w* = 0

  REQUIRES: Dimensionality of a and b must be same

  Args:
    a: A Tensor
    b: A Tensor
    c: A Tensor with one element.

  Returns:
    A Tensor w, which is solution for the equation
  """
  with ops.name_scope("solve_" + b.op.name):
    c = ops.convert_to_tensor(c)
    k = array_ops.fill(array_ops.shape(b), c)
    zero_t = array_ops.zeros(array_ops.shape(b), dtype=b.dtype)
    w = (c * math_ops.sign(b) - b) / a
    w = math_ops.select(math_ops.less(math_ops.abs(b), k), zero_t, w)
    return w


def _Compute(accum, linear, base_lr, lr_power, l1, l2):
  """Compute "variable" given current "accum" and "linear".

  REQUIRES: Dimensionality of accum and linear must be same.

  Args:
    accum: A Tensor which is accumulated gradient square.
    linear: A Tensor with same size of accum.
    base_lr: A Tensor which is base learning rate
    lr_power: A Tensor which is learning rate power
    l1: A Tensor which is l1_regularization strength
    l2: A Tensor which is l2_regularization strength
  Returns:
    A Tensor which is "variable" after update
  """
  with ops.name_scope("compute_" + accum.op.name):
    one_t = constant_op.constant(1.0, dtype=types.float32)
    two_t = constant_op.constant(2.0, dtype=types.float32)
    learning_rate = math_ops.pow(accum, lr_power) * base_lr
    quadratic = one_t / learning_rate + two_t * l2
    w = _Solve(quadratic, linear, l1)
    return w


def _Update(variable, gradients, accum, linear, base_lr, lr_power, l1, l2):
  """Update "variable", "accum", "linear" based on "gradients".

  Some notations here: "variable" as W, "accum" as N, "linear" as Z,
                       "gradients" as G, N(t) means "accum" at t-step.
  Assuming lr_power = -0.5 which means using adagrad learning rate.
  "accum" updates as: N = N + G^2
  "linear" updates as: Z = Z + G - W * (sqrt(N(t)) - sqrt(N(t-1)))/base_lr
  REQUIRES: Dimensionality of variable, gradients, accum and linear
            must be same.

  Args:
    variable: A Variable.
    gradients: A Tensor of same shape as 'variable'.
    accum: A Variable containing the sum of the squares of gradients.
    linear: A Variable containing approximation info.
    base_lr: A constant represents base learning rate.
    lr_power: A constant is used to adjust learning rate.
    l1: A constant represents l1 regularization strength.
    l2: A constant represents l2 regularization strength.

  Returns:
    A group op including three Assign ops:
      1. Assign for "accum"
      2. Assign for "linear"
      3. Assign for "variable"
  """
  dtype = variable.dtype.base_dtype
  base_lr = ops.convert_to_tensor(base_lr, dtype=dtype)
  lr_power = ops.convert_to_tensor(lr_power, dtype=dtype)
  l1 = ops.convert_to_tensor(l1, dtype=dtype)
  l2 = ops.convert_to_tensor(l2, dtype=dtype)
  # Compute the new accumulator
  sqr_grad = math_ops.square(gradients)
  accum_updated = sqr_grad + accum
  # Compute the new linear
  neg_lr_power = math_ops.neg(lr_power)
  sigma = math_ops.pow(accum_updated, neg_lr_power) - math_ops.pow(
      accum, neg_lr_power)
  sigma /= base_lr
  proximal_adjust = sigma * variable
  linear_updated = linear + gradients - proximal_adjust
  # Compute the "variable"
  variable_updated = _Compute(accum_updated, linear_updated, base_lr,
                              lr_power, l1, l2)

  with ops.control_dependencies([sigma]):
    accum_update_op = state_ops.assign(accum, accum_updated)
  linear_update_op = state_ops.assign(linear, linear_updated)
  variable_update_op = state_ops.assign(variable, variable_updated)
  group_op = control_flow_ops.group(linear_update_op, accum_update_op,
                                    variable_update_op)
  return group_op


# TODO(xbing): Refactor code to make _SparseUpdate and _Update share
# common routines.
def _SparseUpdate(variable, gradients, accum, linear, base_lr,
                  lr_power, l1, l2):
  """Sparse Update "variable", "accum", "linear" based on sparse "gradients".

  See the description in _Update.

  Args:
    variable: A Variable.
    gradients: A Sparse Tensor
    accum: A Variable containing the sum of the squares of gradients.
    linear: A Variable containing approximation info.
    base_lr: A constant represents base learning rate.
    lr_power: A constant is used to adjust learning rate.
    l1: A constant represents l1 regularization strength.
    l2: A constant represents l2 regularization strength.

  Returns:
    A group op including three ScatterUpdate ops:
      1. ScatterUpdate for "accum"
      2. ScatterUpdate for "linear"
      3. ScatterUpdate for "variable"
  """
  assert isinstance(gradients, ops.IndexedSlices)
  with ops.name_scope("sparse_update_" + variable.op.name) as scope:
    dtype = variable.dtype.base_dtype
    base_lr = ops.convert_to_tensor(base_lr, dtype=dtype)
    lr_power = ops.convert_to_tensor(lr_power, dtype=dtype)
    l1 = ops.convert_to_tensor(l1, dtype=dtype)
    l2 = ops.convert_to_tensor(l2, dtype=dtype)

    # Compute the new value for the accumulator
    previous_accum = array_ops.gather(accum, gradients.indices)
    sqr_grad = gradients.values * gradients.values
    accum_updated = sqr_grad + previous_accum

    # Compute the new linear
    neg_lr_power = math_ops.neg(lr_power)
    sigma = math_ops.pow(accum_updated, neg_lr_power) - math_ops.pow(
        previous_accum, neg_lr_power)
    sigma /= base_lr
    variable_slice = array_ops.gather(variable, gradients.indices)
    proximal_adjust = sigma * variable_slice
    linear_slice = array_ops.gather(linear, gradients.indices)
    linear_updated = linear_slice + gradients.values - proximal_adjust

    # Compute the new "variable"
    variable_updated = _Compute(accum_updated, linear_updated, base_lr,
                                lr_power, l1, l2)

    with ops.control_dependencies([sigma]):
      accum_update_op = state_ops.scatter_update(accum, gradients.indices,
                                                accum_updated)
    linear_update_op = state_ops.scatter_update(linear, gradients.indices,
                                               linear_updated)
    variable_update_op = state_ops.scatter_update(variable, gradients.indices,
                                                 variable_updated)
    group_op = control_flow_ops.group(linear_update_op, accum_update_op,
                                      variable_update_op, name=scope)
    return group_op


class FtrlOptimizer(optimizer.Optimizer):
  """Optimizer that implements the FTRL algorithm.

  @@__init__
  """

  def __init__(self, learning_rate,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               use_locking=False, name="Ftrl"):
    """Construct a new FTRL optimizer.

    The Ftrl-proximal algorithm, abbreviated for Follow-the-regularized-leader,
    is described in the paper [Ad Click Prediction: a View from the Trenches](
    https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).

    It can give a good performance vs. sparsity tradeoff.

    Ftrl-proximal uses its own global base learning rate and can behave like
    Adagrad with `learning_rate_power=-0.5`, or like gradient descent with
    `learning_rate_power=0.0`.

    The effective learning rate is adjusted per parameter, relative to this
    base learning rate as:

    ```
    effective_learning_rate_i = (learning_rate /
        pow(k + summed_squared_gradients_for_i, learning_rate_power));
    ```

    where k is the small constant `initial_accumulator_value`.

    Note that the real regularization coefficient of `|w|^2` for objective
    function is `1 / lambda_2` if specifying `l2 = lambda_2` as argument when
    using this function.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
      initial_accumulator_value: The starting value for accumulators.
        Only positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Ftrl".

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    super(FtrlOptimizer, self).__init__(use_locking, name)

    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value %f needs to be positive" %
                       initial_accumulator_value)
    if learning_rate_power > 0.0:
      raise ValueError("learning_rate_power %f needs to be negative or zero" %
                       learning_rate_power)
    if l1_regularization_strength < 0.0:
      raise ValueError(
          "l1_regularization_strength %f needs to be positive or zero" %
          l1_regularization_strength)
    if l2_regularization_strength < 0.0:
      raise ValueError(
          "l2_regularization_strength %f needs to be positive or zero" %
          l2_regularization_strength)

    self._learning_rate = learning_rate
    self._learning_rate_power = learning_rate_power
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    for v in var_list:
      self._get_or_make_slot(
          v,
          constant_op.constant(self._initial_accumulator_value,
                               dtype=v.dtype, shape=v.get_shape()),
          "accum",
          self._name)
      self._zeros_slot(v, "linear", self._name)

  def _apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return _Update(var, grad, accum, linear,
                   self._learning_rate, self._learning_rate_power,
                   self._l1_regularization_strength,
                   self._l2_regularization_strength)

  def _apply_sparse(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return _SparseUpdate(var, grad, accum, linear,
                         self._learning_rate, self._learning_rate_power,
                         self._l1_regularization_strength,
                         self._l2_regularization_strength)
