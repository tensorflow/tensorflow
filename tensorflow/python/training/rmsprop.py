"""One-line documentation for rmsprop module.

rmsprop algorithm [tieleman2012rmsprop]

A detailed description of rmsprop.

- maintain a moving (discounted) average of the square of gradients
- divide gradient by the root of this average

mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(mean_square + epsilon)
delta = - mom

"""

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class RMSPropOptimizer(optimizer.Optimizer):
  """Optimizer that implements the RMSProp algorithm.

  @@__init__
  """

  def __init__(self, learning_rate, decay, momentum=0.0, epsilon=1e-10,
               use_locking=False, name="RMSProp"):
    """Construct a new RMSProp optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: discounting factor for the history/coming gradient
      momentum: a scalar tensor.
      epsilon: small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      name: Optional name prefic for the operations created when applying
        gradients. Defaults to "RMSProp".
    """
    super(RMSPropOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      self._get_or_make_slot(
          v, constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape()),
          "rms", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._decay_tensor = ops.convert_to_tensor(self._decay, name="decay")
    self._momentum_tensor = ops.convert_to_tensor(self._momentum,
                                                  name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(self._epsilon,
                                                 name="epsilon")

  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_rms_prop(
        var, rms, mom,
        self._learning_rate_tensor,
        self._decay_tensor,
        self._momentum_tensor,
        self._epsilon_tensor,
        grad, use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()
