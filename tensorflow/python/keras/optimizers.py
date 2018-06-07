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
"""Built-in optimizer classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import optimizer as tf_optimizer_module
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util.tf_export import tf_export


def clip_norm(g, c, n):
  """Clip a tensor by norm.

  Arguments:
    g: gradient tensor to clip.
    c: clipping threshold.
    n: norm of gradient tensor.

  Returns:
    Clipped gradient tensor.
  """
  if c > 0:
    condition = n >= c
    then_expression = lambda: math_ops.scalar_mul(c / n, g)
    else_expression = lambda: g

    # saving the shape to avoid converting sparse tensor to dense
    if isinstance(g, ops.Tensor):
      g_shape = copy.copy(g.get_shape())
    elif isinstance(g, ops.IndexedSlices):
      g_shape = copy.copy(g.dense_shape)
    if condition.dtype != dtypes_module.bool:
      condition = math_ops.cast(condition, 'bool')
    g = control_flow_ops.cond(condition, then_expression, else_expression)
    if isinstance(g, ops.Tensor):
      g.set_shape(g_shape)
    elif isinstance(g, ops.IndexedSlices):
      g._dense_shape = g_shape  # pylint: disable=protected-access
  return g


@tf_export('keras.optimizers.Optimizer')
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
    self.__dict__.update(kwargs)
    self.updates = []
    self.weights = []

  def get_updates(self, loss, params):
    raise NotImplementedError

  def get_gradients(self, loss, params):
    """Returns gradients of `loss` with respect to `params`.

    Arguments:
        loss: Loss tensor.
        params: List of variables.

    Returns:
        List of gradient tensors.

    Raises:
        ValueError: In case any gradient cannot be computed (e.g. if gradient
          function not implemented).
    """
    grads = K.gradients(loss, params)
    if None in grads:
      raise ValueError('An operation has `None` for gradient. '
                       'Please make sure that all of your ops have a '
                       'gradient defined (i.e. are differentiable). '
                       'Common ops without gradient: '
                       'K.argmax, K.round, K.eval.')
    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
      norm = K.sqrt(
          sum([math_ops.reduce_sum(math_ops.square(g)) for g in grads]))
      grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
    if hasattr(self, 'clipvalue') and self.clipvalue > 0:
      grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
    return grads

  def set_weights(self, weights):
    """Sets the weights of the optimizer, from Numpy arrays.

    Should only be called after computing the gradients
    (otherwise the optimizer has no weights).

    Arguments:
        weights: a list of Numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the optimizer (i.e. it should match the
            output of `get_weights`).

    Raises:
        ValueError: in case of incompatible weight shapes.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError(
          'Length of the specified weight list (' + str(len(weights)) +
          ') does not match the number of weights '
          'of the optimizer (' + str(len(params)) + ')')
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError(
            'Optimizer weight shape ' + str(pv.shape) + ' not compatible with '
            'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    K.batch_set_value(weight_value_tuples)

  def get_weights(self):
    """Returns the current value of the weights of the optimizer.

    Returns:
        A list of numpy arrays.
    """
    return K.batch_get_value(self.weights)

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


@tf_export('keras.optimizers.SGD')
class SGD(Optimizer):
  """Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates SGD
          in the relevant direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  """

  def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
    super(SGD, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.momentum = K.variable(momentum, name='momentum')
      self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    self.nesterov = nesterov

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))
    # momentum
    shapes = [K.int_shape(p) for p in params]
    moments = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + moments
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
        'lr': float(K.get_value(self.lr)),
        'momentum': float(K.get_value(self.momentum)),
        'decay': float(K.get_value(self.decay)),
        'nesterov': self.nesterov
    }
    base_config = super(SGD, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.optimizers.RMSprop')
class RMSprop(Optimizer):
  """RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  This optimizer is usually a good choice for recurrent
  neural networks.

  Arguments:
      lr: float >= 0. Learning rate.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  """

  def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., **kwargs):
    super(RMSprop, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.lr = K.variable(lr, name='lr')
      self.rho = K.variable(rho, name='rho')
      self.decay = K.variable(decay, name='decay')
      self.iterations = K.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    self.weights = accumulators
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    for p, g, a in zip(params, grads, accumulators):
      # update accumulator
      new_a = self.rho * a + (1. - self.rho) * math_ops.square(g)
      self.updates.append(state_ops.assign(a, new_a))
      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'rho': float(K.get_value(self.rho)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(RMSprop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.optimizers.Adagrad')
class Adagrad(Optimizer):
  """Adagrad optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
      epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  """

  def __init__(self, lr=0.01, epsilon=None, decay=0., **kwargs):
    super(Adagrad, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.lr = K.variable(lr, name='lr')
      self.decay = K.variable(decay, name='decay')
      self.iterations = K.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    self.weights = accumulators
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    for p, g, a in zip(params, grads, accumulators):
      new_a = a + math_ops.square(g)  # update accumulator
      self.updates.append(state_ops.assign(a, new_a))
      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adagrad, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.optimizers.Adadelta')
class Adadelta(Optimizer):
  """Adadelta optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
          It is recommended to leave it at the default value.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  """

  def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0., **kwargs):
    super(Adadelta, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.lr = K.variable(lr, name='lr')
      self.decay = K.variable(decay, name='decay')
      self.iterations = K.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = K.epsilon()
    self.rho = rho
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    delta_accumulators = [K.zeros(shape) for shape in shapes]
    self.weights = accumulators + delta_accumulators
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
      # update accumulator
      new_a = self.rho * a + (1. - self.rho) * math_ops.square(g)
      self.updates.append(state_ops.assign(a, new_a))

      # use the new accumulator and the *old* delta_accumulator
      update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)
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
        'lr': float(K.get_value(self.lr)),
        'rho': self.rho,
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adadelta, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.optimizers.Adam')
class Adam(Optimizer):
  """Adam optimizer.

  Default parameters follow those provided in the original paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
      amsgrad: boolean. Whether to apply the AMSGrad variant of this
          algorithm from the paper "On the Convergence of Adam and
          Beyond".

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
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
      self.decay = K.variable(decay, name='decay')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay
    self.amsgrad = amsgrad

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    t = math_ops.cast(self.iterations, K.floatx()) + 1
    lr_t = lr * (
        K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
        (1. - math_ops.pow(self.beta_1, t)))

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    if self.amsgrad:
      vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    else:
      vhats = [K.zeros(1) for _ in params]
    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
      if self.amsgrad:
        vhat_t = math_ops.maximum(vhat, v_t)
        p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
        self.updates.append(state_ops.assign(vhat, vhat_t))
      else:
        p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

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
        'lr': float(K.get_value(self.lr)),
        'beta_1': float(K.get_value(self.beta_1)),
        'beta_2': float(K.get_value(self.beta_2)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad
    }
    base_config = super(Adam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.optimizers.Adamax')
class Adamax(Optimizer):
  """Adamax optimizer from Adam paper's Section 7.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
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
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
      self.decay = K.variable(decay, name='decay')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    t = math_ops.cast(self.iterations, K.floatx()) + 1
    lr_t = lr / (1. - math_ops.pow(self.beta_1, t))

    shapes = [K.int_shape(p) for p in params]
    # zero init of 1st moment
    ms = [K.zeros(shape) for shape in shapes]
    # zero init of exponentially weighted infinity norm
    us = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + ms + us

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
        'lr': float(K.get_value(self.lr)),
        'beta_1': float(K.get_value(self.beta_1)),
        'beta_2': float(K.get_value(self.beta_2)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adamax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.optimizers.Nadam')
class Nadam(Optimizer):
  """Nesterov Adam optimizer.

  Much like Adam is essentially RMSprop with momentum,
  Nadam is Adam RMSprop with Nesterov momentum.

  Default parameters follow those provided in the paper.
  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

  """

  def __init__(self,
               lr=0.002,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               schedule_decay=0.004,
               **kwargs):
    super(Nadam, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.m_schedule = K.variable(1., name='m_schedule')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.schedule_decay = schedule_decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    t = math_ops.cast(self.iterations, K.floatx()) + 1

    # Due to the recommendations in [2], i.e. warming momentum schedule
    momentum_cache_t = self.beta_1 * (
        1. - 0.5 *
        (math_ops.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
    momentum_cache_t_1 = self.beta_1 * (
        1. - 0.5 *
        (math_ops.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
    m_schedule_new = self.m_schedule * momentum_cache_t
    m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
    self.updates.append((self.m_schedule, m_schedule_new))

    shapes = [K.int_shape(p) for p in params]
    ms = [K.zeros(shape) for shape in shapes]
    vs = [K.zeros(shape) for shape in shapes]

    self.weights = [self.iterations] + ms + vs

    for p, g, m, v in zip(params, grads, ms, vs):
      # the following equations given in [1]
      g_prime = g / (1. - m_schedule_new)
      m_t = self.beta_1 * m + (1. - self.beta_1) * g
      m_t_prime = m_t / (1. - m_schedule_next)
      v_t = self.beta_2 * v + (1. - self.beta_2) * math_ops.square(g)
      v_t_prime = v_t / (1. - math_ops.pow(self.beta_2, t))
      m_t_bar = (
          1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

      self.updates.append(state_ops.assign(m, m_t))
      self.updates.append(state_ops.assign(v, v_t))

      p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'beta_1': float(K.get_value(self.beta_1)),
        'beta_2': float(K.get_value(self.beta_2)),
        'epsilon': self.epsilon,
        'schedule_decay': self.schedule_decay
    }
    base_config = super(Nadam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class TFOptimizer(Optimizer, checkpointable.Checkpointable):
  """Wrapper class for native TensorFlow optimizers.
  """

  def __init__(self, optimizer):  # pylint: disable=super-init-not-called
    self.optimizer = optimizer
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')

  def apply_gradients(self, grads):
    self.optimizer.apply_gradients(grads)

  def get_grads(self, loss, params):
    return self.optimizer.compute_gradients(loss, params)

  def get_updates(self, loss, params):
    if distribute_lib.has_distribution_strategy():
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
      self.updates = [state_ops.assign_add(self.iterations, 1)]
      if not params:
        return self.updates

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


@tf_export('keras.optimizers.serialize')
def serialize(optimizer):
  return serialize_keras_object(optimizer)


@tf_export('keras.optimizers.deserialize')
def deserialize(config, custom_objects=None):
  """Inverse of the `serialize` function.

  Arguments:
      config: Optimizer configuration dictionary.
      custom_objects: Optional dictionary mapping
          names (strings) to custom objects
          (classes and functions)
          to be considered during deserialization.

  Returns:
      A Keras Optimizer instance.
  """
  all_classes = {
      'sgd': SGD,
      'rmsprop': RMSprop,
      'adagrad': Adagrad,
      'adadelta': Adadelta,
      'adam': Adam,
      'adamax': Adamax,
      'nadam': Nadam,
      'tfoptimizer': TFOptimizer,
  }
  # Make deserialization case-insensitive for built-in optimizers.
  if config['class_name'].lower() in all_classes:
    config['class_name'] = config['class_name'].lower()
  return deserialize_keras_object(
      config,
      module_objects=all_classes,
      custom_objects=custom_objects,
      printable_module_name='optimizer')


@tf_export('keras.optimizers.get')
def get(identifier):
  """Retrieves a Keras Optimizer instance.

  Arguments:
      identifier: Optimizer identifier, one of
          - String: name of an optimizer
          - Dictionary: configuration dictionary.
          - Keras Optimizer instance (it will be returned unchanged).
          - TensorFlow Optimizer instance
              (it will be wrapped as a Keras Optimizer).

  Returns:
      A Keras Optimizer instance.

  Raises:
      ValueError: If `identifier` cannot be interpreted.
  """
  # Wrap TF optimizer instances
  if isinstance(identifier, tf_optimizer_module.Optimizer):
    return TFOptimizer(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  if isinstance(identifier, Optimizer):
    return identifier
  else:
    raise ValueError('Could not interpret optimizer identifier:', identifier)
