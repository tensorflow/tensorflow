"""Maintain moving averages of parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables


# TODO(touts): switch to variables.Variable.
def assign_moving_average(variable, value, decay, name=None):
  """Compute the moving average of a variable.

  The moving average of 'variable' updated with 'value' is:
    variable * decay + value * (1 - decay)

  The returned Operation sets 'variable' to the newly computed moving average.

  The new value of 'variable' can be set with the 'AssignSub' op as:
     variable -= (1 - decay) * (variable - value)

  Args:
    variable: A Variable.
    value: A tensor with the same shape as 'variable'
    decay: A float Tensor or float value.  The moving average decay.
    name: Optional name of the returned operation.

  Returns:
    An Operation that updates 'variable' with the newly computed
    moving average.
  """
  with ops.op_scope([variable, value, decay], name, "AssignMovingAvg") as name:
    with ops.device(variable.device):
      decay = ops.convert_to_tensor(1.0 - decay, name="decay")
      if decay.dtype != variable.dtype.base_dtype:
        decay = math_ops.cast(decay, variable.dtype.base_dtype)
      return state_ops.assign_sub(variable, (variable - value) * decay,
                                  name=name)


class ExponentialMovingAverage(object):
  """Maintains moving averages of variables by employing an exponential decay.

  When training a model, it is often beneficial to maintain moving averages of
  the trained parameters.  Evaluations that use averaged parameters sometimes
  produce significantly better results than the final trained values.

  The `apply()` method adds shadow copies of trained variables and add ops that
  maintain a moving average of the trained variables in their shadow copies.
  It is used when building the training model.  The ops that maintain moving
  averages are typically run after each training step.
  The `average()` and `average_name()` methods give access to the shadow
  variables and their names.  They are useful when building an evaluation
  model, or when restoring a model from a checkpoint file.  They help use the
  moving averages in place of the last trained values for evaluations.

  The moving averages are computed using exponential decay.  You specify the
  decay value when creating the `ExponentialMovingAverage` object.  The shadow
  variables are initialized with the same initial values as the trained
  variables.  When you run the ops to maintain the moving averages, each
  shadow variable is updated with the formula:

    `shadow_variable -= (1 - decay) * (shadow_variable - variable)`

  This is mathematically equivalent to the classic formula below, but the use
  of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
  updates to the variables:

    `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

  Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.9999, etc.

  Example usage when creating a training model:

  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...
  ...
  # Create an op that applies the optimizer.  This is what we usually
  # would use as a training op.
  opt_op = opt.minimize(my_loss, [var0, var1])

  # Create an ExponentialMovingAverage object
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)

  # Create the shadow variables, and add ops to maintain moving averages
  # of var0 and var1.
  maintain_averages_op = ema.apply([var0, var1])

  # Create an op that will update the moving averages after each training
  # step.  This is what we will use in place of the usuall trainig op.
  with tf.control_dependencies([opt_op]):
      training_op = tf.group(maintain_averages_op)

  ...train the model by running training_op...
  ```

  There are two ways to use the moving averages for evaluations:

  *  Build a model that uses the shadow variables instead of the variables.
     For this, use the `average()` method which returns the shadow variable
     for a given variable.
  *  Build a model normally but load the checkpoint files to evaluate by using
     the shadow variable names.  For this use the `average_name()` method.  See
     the [Saver class](../../api_docs/python/train.md#Saver) for more
     information on restoring saved variables.

  Example of restoring the shadow variable values:

  ```python
  # Create a Saver that loads variables from their saved shadow values.
  shadow_var0_name = ema.average_name(var0)
  shadow_var1_name = ema.average_name(var1)
  saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
  saver.restore(...checkpoint filename...)
  # var0 and var1 now hold the moving average values
  ```

  @@__init__
  @@apply
  @@average_name
  @@average
  """

  def __init__(self, decay, num_updates=None,
               name="ExponentialMovingAverage"):
    """Creates a new ExponentialMovingAverage object.

    The `Apply()` method has to be called to create shadow variables and add
    ops to maintain moving averages.

    The optional `num_updates` parameter allows one to tweak the decay rate
    dynamically. .  It is typical to pass the count of training steps, usually
    kept in a variable that is incremented at each step, in which case the
    decay rate is lower at the start of training.  This makes moving averages
    move faster.  If passed, the actual decay rate used is:

      `min(decay, (1 + num_updates) / (10 + num_updates))`

    Args:
      decay: Float.  The decay to use.
      num_updates: Optional count of number of updates applied to variables.
      name: String. Optional prefix name to use for the name of ops added in
        `Apply()`.
    """
    self._decay = decay
    self._num_updates = num_updates
    self._name = name
    self._averages = {}

  def apply(self, var_list=None):
    """Maintains moving averages of variables.

    `var_list` must be a list of `Variable` or `Tensor` objects.  This method
    creates shadow variables for all elements of `var_list`.  Shadow variables
    for `Variable` objects are initialized to the variable's initial value.
    For `Tensor` objects, the shadow variables are initialized to 0.

    shadow variables are created with `trainable=False` and added to the
    `GraphKeys.ALL_VARIABLES` collection.  They will be returned by calls to
    `tf.all_variables()`.

    Returns an op that updates all shadow variables as described above.

    Note that `apply()` can be called multiple times with different lists of
    variables.

    Args:
      var_list: A list of Variable or Tensor objects. The variables
        and Tensors must be of types float32 or float64.

    Returns:
      An Operation that updates the moving averages.

    Raises:
      TypeError: If the arguments are not all float32 or float64.
      ValueError: If the moving average of one of the variables is already
        being computed.
    """
    # TODO(touts): op_scope
    if var_list is None:
      var_list = variables.trainable_variables()
    for var in var_list:
      if var.dtype.base_dtype not in [types.float32, types.float64]:
        raise TypeError("The variables must be float or double: %s" % var)
      if var in self._averages:
        raise ValueError("Moving average already computed for: %s" % var)
      with ops.name_scope(var.op.name + "/" + self._name) as scope:
        with ops.device(var.device):
          if isinstance(var, variables.Variable):
            initial_value = var.initialized_value()
          else:
            initial_value = array_ops.zeros(var.get_shape().as_list())
          avg = variables.Variable(initial_value, name=scope, trainable=False)
          self._averages[var] = avg
    with ops.name_scope(self._name) as scope:
      decay = ops.convert_to_tensor(self._decay, name="decay")
      if self._num_updates is not None:
        num_updates = math_ops.cast(self._num_updates, types.float32,
                                    name="num_updates")
        decay = math_ops.minimum(decay,
                                 (1.0 + num_updates) / (10.0 + num_updates))
      updates = []
      for var in var_list:
        updates.append(assign_moving_average(self._averages[var], var, decay))
      return control_flow_ops.group(*updates, name=scope)

  def average(self, var):
    """Returns the `Variable` holding the average of `var`.

    Args:
      var: A `Variable` object.

    Returns:
      A `Variable` object or `None` if the moving average of `var`
      is not maintained..
    """
    return self._averages.get(var, None)

  def average_name(self, var):
    """Returns the name of the `Variable` holding the average for `var`.

    The typical scenario for `ExponentialMovingAverage` is to compute moving
    averages of variables during training, and restore the variables from the
    computed moving averages during evaluations.

    To restore variables, you have to know the name of the shadow variables.
    That name and the original variable can then be passed to a `Saver()` object
    to restore the variable from the moving average value with:
      `saver = tf.train.Saver({ema.average_name(var): var})`

    `average_name()` can be called whether or not `apply()` has been called.

    Args:
      var: A `Variable` object.

    Returns:
      A string: the name of the variable that will be used or was used
      by the `ExponentialMovingAverage class` to hold the moving average of
      `var`.
    """
    return var.op.name + "/" + self._name
