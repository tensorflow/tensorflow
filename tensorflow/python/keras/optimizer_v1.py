# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
# pylint: disable=g-classes-have-attributes
"""Legacy v1 optimizer classes.

For more examples see the base class `tf.compat.v1.keras.optimizers.Optimizer`.
"""

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_util
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest


class Optimizer(object):
  """Abstract optimizer base class.

  Note: this is the parent class of all optimizers, not an actual optimizer
  that can be used for training models.

  All Keras optimizers support the following keyword arguments:

      clipnorm: float >= 0. Gradients will be clipped
          when their L2 norm exceeds this value.
      clipvalue: float >= 0. Gradients will be clipped
          when their absolute value exceeds this value.
  """

  def __init__(self, **kwargs):
    allowed_kwargs = {'clipnorm', 'clipvalue'}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError('Unexpected keyword argument '
                        'passed to optimizer: ' + str(k))
      # checks that clipnorm >= 0 and clipvalue >= 0
      if kwargs[k] < 0:
        raise ValueError('Expected {} >= 0, received: {}'.format(k, kwargs[k]))
    self.__dict__.update(kwargs)
    self.updates = []
    self.weights = []

  # Set this to False, indicating `apply_gradients` does not take the
  # `experimental_aggregate_gradients` argument.
  _HAS_AGGREGATE_GRAD = False

  def _create_all_weights(self, params):
    """Creates and sets all optimizer weights.

    Args:
      params: list or tuple of `Variable` objects that will be minimized
        using this optimizer.

    Returns:
      Specific weight values that are used in `get_updates`
    """
    raise NotImplementedError

  def get_updates(self, loss, params):
    raise NotImplementedError

  def get_gradients(self, loss, params):
    """Returns gradients of `loss` with respect to `params`.

    Args:
        loss: Loss tensor.
        params: List of variables.

    Returns:
        List of gradient tensors.

    Raises:
        ValueError: In case any gradient cannot be computed (e.g. if gradient
          function not implemented).
    """
    grads = backend.gradients(loss, params)
    if any(g is None for g in grads):
      raise ValueError('An operation has `None` for gradient. '
                       'Please make sure that all of your ops have a '
                       'gradient defined (i.e. are differentiable). '
                       'Common ops without gradient: '
                       'backend.argmax, backend.round, backend.eval.')
    if hasattr(self, 'clipnorm'):
      grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
    if hasattr(self, 'clipvalue'):
      grads = [
          clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
          for g in grads
      ]
    return grads

  def set_weights(self, weights):
    """Sets the weights of the optimizer, from Numpy arrays.

    Should only be called after computing the gradients
    (otherwise the optimizer has no weights).

    Args:
        weights: a list of Numpy arrays. The number of arrays and their shape
          must match number of the dimensions of the weights of the optimizer
          (i.e. it should match the output of `get_weights`).

    Raises:
        ValueError: in case of incompatible weight shapes.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError('Length of the specified weight list (' +
                       str(len(weights)) +
                       ') does not match the number of weights '
                       'of the optimizer (' + str(len(params)) + ')')
    weight_value_tuples = []
    param_values = backend.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Optimizer weight shape ' + str(pv.shape) +
                         ' not compatible with '
                         'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    backend.batch_set_value(weight_value_tuples)

  def get_weights(self):
    """Returns the current value of the weights of the optimizer.

    Returns:
        A list of numpy arrays.
    """
    return backend.batch_get_value(self.weights)

  def get_config(self):
    config = {}
    if hasattr(self, 'clipnorm'):
      config['clipnorm'] = self.clipnorm
    if hasattr(self, 'clipvalue'):
      config['clipvalue'] = self.clipvalue
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class SGD(Optimizer):
  """Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Args:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates SGD in the relevant
        direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  """

  def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
    super(SGD, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
      self.lr = backend.variable(lr, name='lr')
      self.momentum = backend.variable(momentum, name='momentum')
      self.decay = backend.variable(decay, name='decay')
    self.initial_decay = decay
    self.nesterov = nesterov

  def _create_all_weights(self, params):
    shapes = [backend.int_shape(p) for p in params]
    moments = [backend.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + moments
    return moments

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. /
          (1. +
           self.decay * math_ops.cast(self.iterations,
                                      backend.dtype(self.decay))))
    # momentum
    moments = self._create_all_weights(params)
    for p, g, m in zip(params, grads, moments):
      v = self.momentum * m - lr * g  # velocity
      self.updates.append(state_ops.assign(m, v))

      if self.nesterov:
        new_p = p + self.momentum * v - lr * g
      else:
        new_p = p + v

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'momentum': float(backend.get_value(self.momentum)),
        'decay': float(backend.get_value(self.decay)),
        'nesterov': self.nesterov
    }
    base_config = super(SGD, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class RMSprop(Optimizer):
  """RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  Args:
    lr: float >= 0. Learning rate.
    rho: float >= 0.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Learning rate decay over each update.
  """

  def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., **kwargs):
    super(RMSprop, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.lr = backend.variable(lr, name='lr')
      self.rho = backend.variable(rho, name='rho')
      self.decay = backend.variable(decay, name='decay')
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = backend.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def _create_all_weights(self, params):
    accumulators = [
        backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
        for p in params]
    self.weights = accumulators
    return accumulators

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    accumulators = self._create_all_weights(params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. /
          (1. +
           self.decay * math_ops.cast(self.iterations,
                                      backend.dtype(self.decay))))

    for p, g, a in zip(params, grads, accumulators):
      # update accumulator
      new_a = self.rho * a + (1. - self.rho) * math_ops.square(g)
      self.updates.append(state_ops.assign(a, new_a))
      new_p = p - lr * g / (backend.sqrt(new_a) + self.epsilon)

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'rho': float(backend.get_value(self.rho)),
        'decay': float(backend.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(RMSprop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Adagrad(Optimizer):
  """Adagrad optimizer.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  # Arguments
      lr: float >= 0. Initial learning rate.
      epsilon: float >= 0. If `None`, defaults to `backend.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  # References
      - [Adaptive Subgradient Methods for Online Learning and Stochastic
      Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  """

  def __init__(self, lr=0.01, epsilon=None, decay=0., **kwargs):
    super(Adagrad, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.lr = backend.variable(lr, name='lr')
      self.decay = backend.variable(decay, name='decay')
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = backend.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def _create_all_weights(self, params):
    shapes = [backend.int_shape(p) for p in params]
    accumulators = [backend.zeros(shape) for shape in shapes]
    self.weights = accumulators
    return accumulators

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    accumulators = self._create_all_weights(params)

    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. /
          (1. +
           self.decay * math_ops.cast(self.iterations,
                                      backend.dtype(self.decay))))

    for p, g, a in zip(params, grads, accumulators):
      new_a = a + math_ops.square(g)  # update accumulator
      self.updates.append(state_ops.assign(a, new_a))
      new_p = p - lr * g / (backend.sqrt(new_a) + self.epsilon)

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'decay': float(backend.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adagrad, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Adadelta(Optimizer):
  """Adadelta optimizer.

  Adadelta is a more robust extension of Adagrad
  that adapts learning rates based on a moving window of gradient updates,
  instead of accumulating all past gradients. This way, Adadelta continues
  learning even when many updates have been done. Compared to Adagrad, in the
  original version of Adadelta you don't have to set an initial learning
  rate. In this version, initial learning rate and decay factor can
  be set, as in most other Keras optimizers.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
    lr: float >= 0. Initial learning rate, defaults to 1.
        It is recommended to leave it at the default value.
    rho: float >= 0. Adadelta decay factor, corresponding to fraction of
        gradient to keep at each time step.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Initial learning rate decay.

  References:
      - [Adadelta - an adaptive learning rate
      method](http://arxiv.org/abs/1212.5701)
  """

  def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0., **kwargs):
    super(Adadelta, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.lr = backend.variable(lr, name='lr')
      self.decay = backend.variable(decay, name='decay')
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = backend.epsilon()
    self.rho = rho
    self.epsilon = epsilon
    self.initial_decay = decay

  def _create_all_weights(self, params):
    shapes = [backend.int_shape(p) for p in params]
    accumulators = [backend.zeros(shape) for shape in shapes]
    delta_accumulators = [backend.zeros(shape) for shape in shapes]
    self.weights = accumulators + delta_accumulators
    return accumulators, delta_accumulators

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]
    accumulators, delta_accumulators = self._create_all_weights(params)

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. /
          (1. +
           self.decay * math_ops.cast(self.iterations,
                                      backend.dtype(self.decay))))

    for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
      # update accumulator
      new_a = self.rho * a + (1. - self.rho) * math_ops.square(g)
      self.updates.append(state_ops.assign(a, new_a))

      # use the new accumulator and the *old* delta_accumulator
      update = g * backend.sqrt(d_a + self.epsilon) / backend.sqrt(
          new_a + self.epsilon)
      new_p = p - lr * update

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))

      # update delta_accumulator
      new_d_a = self.rho * d_a + (1 - self.rho) * math_ops.square(update)
      self.updates.append(state_ops.assign(d_a, new_d_a))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'rho': self.rho,
        'decay': float(backend.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adadelta, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Adam(Optimizer):
  """Adam optimizer.

  Default parameters follow those provided in the original paper.

  Args:
    lr: float >= 0. Learning rate.
    beta_1: float, 0 < beta < 1. Generally close to 1.
    beta_2: float, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Learning rate decay over each update.
    amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm
      from the paper "On the Convergence of Adam and Beyond".
  """

  def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               decay=0.,
               amsgrad=False,
               **kwargs):
    super(Adam, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
      self.lr = backend.variable(lr, name='lr')
      self.beta_1 = backend.variable(beta_1, name='beta_1')
      self.beta_2 = backend.variable(beta_2, name='beta_2')
      self.decay = backend.variable(decay, name='decay')
    if epsilon is None:
      epsilon = backend.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay
    self.amsgrad = amsgrad

  def _create_all_weights(self, params):
    ms = [
        backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
        for p in params]
    vs = [
        backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
        for p in params]
    if self.amsgrad:
      vhats = [
          backend.zeros(backend.int_shape(p), dtype=backend.dtype(p))
          for p in params]
    else:
      vhats = [backend.zeros(1) for _ in params]
    self.weights = [self.iterations] + ms + vs + vhats
    return ms, vs, vhats

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = []

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. /
          (1. +
           self.decay * math_ops.cast(self.iterations,
                                      backend.dtype(self.decay))))

    with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
      t = math_ops.cast(self.iterations, backend.floatx())
    lr_t = lr * (
        backend.sqrt(1. - math_ops.pow(self.beta_2, t)) /
        (1. - math_ops.pow(self.beta_1, t)))

    ms, vs, vhats = self._create_all_weights(params)
    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
      if self.amsgrad:
        vhat_t = math_ops.maximum(vhat, v_t)
        p_t = p - lr_t * m_t / (backend.sqrt(vhat_t) + self.epsilon)
        self.updates.append(state_ops.assign(vhat, vhat_t))
      else:
        p_t = p - lr_t * m_t / (backend.sqrt(v_t) + self.epsilon)

      self.updates.append(state_ops.assign(m, m_t))
      self.updates.append(state_ops.assign(v, v_t))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'beta_1': float(backend.get_value(self.beta_1)),
        'beta_2': float(backend.get_value(self.beta_2)),
        'decay': float(backend.get_value(self.decay)),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad
    }
    base_config = super(Adam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Adamax(Optimizer):
  """Adamax optimizer from Adam paper's Section 7.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.

  Args:
    lr: float >= 0. Learning rate.
    beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Learning rate decay over each update.
  """

  def __init__(self,
               lr=0.002,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               decay=0.,
               **kwargs):
    super(Adamax, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
      self.lr = backend.variable(lr, name='lr')
      self.beta_1 = backend.variable(beta_1, name='beta_1')
      self.beta_2 = backend.variable(beta_2, name='beta_2')
      self.decay = backend.variable(decay, name='decay')
    if epsilon is None:
      epsilon = backend.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def _create_all_weights(self, params):

    shapes = [backend.int_shape(p) for p in params]
    # zero init of 1st moment
    ms = [backend.zeros(shape) for shape in shapes]
    # zero init of exponentially weighted infinity norm
    us = [backend.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + ms + us
    return ms, us

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = []

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. /
          (1. +
           self.decay * math_ops.cast(self.iterations,
                                      backend.dtype(self.decay))))

    with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
      t = math_ops.cast(self.iterations, backend.floatx())
    lr_t = lr / (1. - math_ops.pow(self.beta_1, t))

    ms, us = self._create_all_weights(params)

    for p, g, m, u in zip(params, grads, ms, us):

      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      u_t = math_ops.maximum(self.beta_2 * u, math_ops.abs(g))
      p_t = p - lr_t * m_t / (u_t + self.epsilon)

      self.updates.append(state_ops.assign(m, m_t))
      self.updates.append(state_ops.assign(u, u_t))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'beta_1': float(backend.get_value(self.beta_1)),
        'beta_2': float(backend.get_value(self.beta_2)),
        'decay': float(backend.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adamax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Nadam(Optimizer):
  """Nesterov Adam optimizer.

  Much like Adam is essentially RMSprop with momentum,
  Nadam is Adam RMSprop with Nesterov momentum.

  Default parameters follow those provided in the paper.
  It is recommended to leave the parameters of this optimizer
  at their default values.

  Args:
    lr: float >= 0. Learning rate.
    beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
  """

  def __init__(self,
               lr=0.002,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               schedule_decay=0.004,
               **kwargs):
    super(Nadam, self).__init__(**kwargs)
    with backend.name_scope(self.__class__.__name__):
      self.iterations = backend.variable(0, dtype='int64', name='iterations')
      self.m_schedule = backend.variable(1., name='m_schedule')
      self.lr = backend.variable(lr, name='lr')
      self.beta_1 = backend.variable(beta_1, name='beta_1')
      self.beta_2 = backend.variable(beta_2, name='beta_2')
    if epsilon is None:
      epsilon = backend.epsilon()
    self.epsilon = epsilon
    self.schedule_decay = schedule_decay

  def _create_all_weights(self, params):
    shapes = [backend.int_shape(p) for p in params]
    ms = [backend.zeros(shape) for shape in shapes]
    vs = [backend.zeros(shape) for shape in shapes]

    self.weights = [self.iterations, self.m_schedule] + ms + vs
    return ms, vs

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = []

    with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
      t = math_ops.cast(self.iterations, backend.floatx())

    # Due to the recommendations in [2], i.e. warming momentum schedule
    momentum_cache_t = self.beta_1 * (
        1. - 0.5 *
        (math_ops.pow(backend.cast_to_floatx(0.96), t * self.schedule_decay)))
    momentum_cache_t_1 = self.beta_1 * (
        1. - 0.5 *
        (math_ops.pow(backend.cast_to_floatx(0.96),
                      (t + 1) * self.schedule_decay)))
    m_schedule_new = self.m_schedule * momentum_cache_t
    m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
    self.updates.append((self.m_schedule, m_schedule_new))

    ms, vs = self._create_all_weights(params)

    for p, g, m, v in zip(params, grads, ms, vs):
      # the following equations given in [1]
      g_prime = g / (1. - m_schedule_new)
      m_t = self.beta_1 * m + (1. - self.beta_1) * g
      m_t_prime = m_t / (1. - m_schedule_next)
      v_t = self.beta_2 * v + (1. - self.beta_2) * math_ops.square(g)
      v_t_prime = v_t / (1. - math_ops.pow(self.beta_2, t))
      m_t_bar = (1. -
                 momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

      self.updates.append(state_ops.assign(m, m_t))
      self.updates.append(state_ops.assign(v, v_t))

      p_t = p - self.lr * m_t_bar / (backend.sqrt(v_t_prime) + self.epsilon)
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(backend.get_value(self.lr)),
        'beta_1': float(backend.get_value(self.beta_1)),
        'beta_2': float(backend.get_value(self.beta_2)),
        'epsilon': self.epsilon,
        'schedule_decay': self.schedule_decay
    }
    base_config = super(Nadam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class TFOptimizer(Optimizer, trackable.Trackable):
  """Wrapper class for native TensorFlow optimizers."""

  def __init__(self, optimizer, iterations=None):  # pylint: disable=super-init-not-called
    self.optimizer = optimizer
    self._track_trackable(optimizer, name='optimizer')
    if iterations is None:
      with backend.name_scope(self.__class__.__name__):
        self.iterations = backend.variable(0, dtype='int64', name='iterations')
    else:
      self.iterations = iterations
    self._track_trackable(self.iterations, name='global_step')

  def _clip_gradients(self, grads):
    """Clip gradients according to the clipnorm and clipvalue attributes."""
    # TFOptimizer wrapper has no gradient clipping options.
    return grads

  def minimize(self, loss, var_list, grad_loss=None, tape=None):
    """Mimics the `OptimizerV2.minimize` API."""
    if not callable(loss) and tape is None:
      raise ValueError('`tape` is required when a `Tensor` loss is passed.')
    tape = tape if tape is not None else backprop.GradientTape()

    if callable(loss):
      with tape:
        if not callable(var_list):
          tape.watch(var_list)
        loss = loss()
        if callable(var_list):
          var_list = var_list()

    var_list = nest.flatten(var_list)
    if var_list:
      grads = tape.gradient(loss, var_list, grad_loss)
      grads_and_vars = list(zip(grads, var_list))
      self.apply_gradients(grads_and_vars)

  def apply_gradients(self, grads_and_vars):
    self.optimizer.apply_gradients(grads_and_vars, global_step=self.iterations)

  def get_grads(self, loss, params):
    return self.optimizer.compute_gradients(loss, params)

  def get_updates(self, loss, params):
    if distribution_strategy_context.has_strategy():
      self.updates = []

      if not params:
        # After the model vars have been created, the second call to get_updates
        # is called with params as an empty list. This ensures that we call
        # compute_gradients with params=None.
        grads = self.optimizer.compute_gradients(loss)
      else:
        grads = self.optimizer.compute_gradients(loss, params)
      global_step = training_util.get_global_step()
      opt_update = self.optimizer.apply_gradients(grads, global_step)
    else:
      if not params:
        self.updates = [state_ops.assign_add(self.iterations, 1)]
        return self.updates

      # Updates list starts out empty because the iterations variable is
      # incremented in optimizer.apply_gradients()
      self.updates = []
      grads = self.optimizer.compute_gradients(loss, params)
      opt_update = self.optimizer.apply_gradients(
          grads, global_step=self.iterations)

    self.updates.append(opt_update)
    return self.updates

  @property
  def weights(self):
    raise NotImplementedError

  def get_config(self):
    raise NotImplementedError

  def from_config(self, config):
    raise NotImplementedError


# Aliases.

sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax
nadam = Nadam
