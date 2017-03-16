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
"""Keras optimizer classes (will eventually be replaced with core optimizers).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.training import optimizer as tf_optimizer_module


def clip_norm(g, c, n):
  if c > 0:
    g = K.switch(n >= c, g * c / n, g)
  return g


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

  def get_updates(self, params, constraints, loss):
    raise NotImplementedError

  def get_gradients(self, loss, params):
    grads = K.gradients(loss, params)
    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
      norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
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
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Optimizer weight shape ' + str(pv.shape) +
                         ' not compatible with '
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


class SGD(Optimizer):
  """Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter updates momentum.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  """

  def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
    super(SGD, self).__init__(**kwargs)
    self.iterations = K.variable(0., name='iterations')
    self.lr = K.variable(lr, name='lr')
    self.momentum = K.variable(momentum, name='momentum')
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    self.nesterov = nesterov

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    self.updates = []

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))
      self.updates.append(K.update_add(self.iterations, 1))

    # momentum
    shapes = [K.int_shape(p) for p in params]
    moments = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + moments
    for p, g, m in zip(params, grads, moments):
      v = self.momentum * m - lr * g  # velocity
      self.updates.append(K.update(m, v))

      if self.nesterov:
        new_p = p + self.momentum * v - lr * g
      else:
        new_p = p + v

      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)

      self.updates.append(K.update(p, new_p))
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


class RMSprop(Optimizer):
  # pylint: disable=line-too-long
  """RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  This optimizer is usually a good choice for recurrent
  neural networks.

  Arguments:
      lr: float >= 0. Learning rate.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor.
      decay: float >= 0. Learning rate decay over each update.

  References:
      - [rmsprop: Divide the gradient by a running average of its recent
        magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
  """

  # pylint: enable=line-too-long

  def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, decay=0., **kwargs):
    super(RMSprop, self).__init__(**kwargs)
    self.lr = K.variable(lr, name='lr')
    self.rho = K.variable(rho, name='rho')
    self.epsilon = epsilon
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    self.iterations = K.variable(0., name='iterations')

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    self.weights = accumulators
    self.updates = []

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))
      self.updates.append(K.update_add(self.iterations, 1))

    for p, g, a in zip(params, grads, accumulators):
      # update accumulator
      new_a = self.rho * a + (1. - self.rho) * K.square(g)
      self.updates.append(K.update(a, new_a))
      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)
      self.updates.append(K.update(p, new_p))
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


class Adagrad(Optimizer):
  # pylint: disable=line-too-long
  """Adagrad optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
      epsilon: float >= 0.
      decay: float >= 0. Learning rate decay over each update.

  References:
      - [Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  """

  # pylint: enable=line-too-long

  def __init__(self, lr=0.01, epsilon=1e-8, decay=0., **kwargs):
    super(Adagrad, self).__init__(**kwargs)
    self.lr = K.variable(lr, name='lr')
    self.epsilon = epsilon
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    self.iterations = K.variable(0., name='iterations')

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    self.weights = accumulators
    self.updates = []

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))
      self.updates.append(K.update_add(self.iterations, 1))

    for p, g, a in zip(params, grads, accumulators):
      new_a = a + K.square(g)  # update accumulator
      self.updates.append(K.update(a, new_a))
      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)
      self.updates.append(K.update(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adagrad, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Adadelta(Optimizer):
  # pylint: disable=line-too-long
  """Adadelta optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
          It is recommended to leave it at the default value.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor.
      decay: float >= 0. Learning rate decay over each update.

  References:
      - [Adadelta - an adaptive learning rate
        method](http://arxiv.org/abs/1212.5701)
  """

  # pylint: enable=line-too-long

  def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8, decay=0., **kwargs):
    super(Adadelta, self).__init__(**kwargs)
    self.lr = K.variable(lr, name='lr')
    self.rho = rho
    self.epsilon = epsilon
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay
    self.iterations = K.variable(0., name='iterations')

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    delta_accumulators = [K.zeros(shape) for shape in shapes]
    self.weights = accumulators + delta_accumulators
    self.updates = []

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))
      self.updates.append(K.update_add(self.iterations, 1))

    for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
      # update accumulator
      new_a = self.rho * a + (1. - self.rho) * K.square(g)
      self.updates.append(K.update(a, new_a))

      # use the new accumulator and the *old* delta_accumulator
      update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)

      new_p = p - lr * update
      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)
      self.updates.append(K.update(p, new_p))

      # update delta_accumulator
      new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
      self.updates.append(K.update(d_a, new_d_a))
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


class Adam(Optimizer):
  # pylint: disable=line-too-long
  """Adam optimizer.

  Default parameters follow those provided in the original paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor.
      decay: float >= 0. Learning rate decay over each update.

  References:
      - [Adam - A Method for Stochastic
        Optimization](http://arxiv.org/abs/1412.6980v8)
  """

  # pylint: enable=line-too-long

  def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               decay=0.,
               **kwargs):
    super(Adam, self).__init__(**kwargs)
    self.iterations = K.variable(0, name='iterations')
    self.lr = K.variable(lr, name='lr')
    self.beta_1 = K.variable(beta_1, name='beta_1')
    self.beta_2 = K.variable(beta_2, name='beta_2')
    self.epsilon = epsilon
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))

    t = self.iterations + 1
    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                 (1. - K.pow(self.beta_1, t)))

    shapes = [K.int_shape(p) for p in params]
    ms = [K.zeros(shape) for shape in shapes]
    vs = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + ms + vs

    for p, g, m, v in zip(params, grads, ms, vs):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))

      new_p = p_t
      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)
      self.updates.append(K.update(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'beta_1': float(K.get_value(self.beta_1)),
        'beta_2': float(K.get_value(self.beta_2)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(Adam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Adamax(Optimizer):
  # pylint: disable=line-too-long
  """Adamax optimizer from Adam paper's Section 7.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor.
      decay: float >= 0. Learning rate decay over each update.

  References:
      - [Adam - A Method for Stochastic
        Optimization](http://arxiv.org/abs/1412.6980v8)
  """

  # pylint: enable=line-too-long

  def __init__(self,
               lr=0.002,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               decay=0.,
               **kwargs):
    super(Adamax, self).__init__(**kwargs)
    self.iterations = K.variable(0., name='iterations')
    self.lr = K.variable(lr, name='lr')
    self.beta_1 = K.variable(beta_1, name='beta_1')
    self.beta_2 = K.variable(beta_2, name='beta_2')
    self.epsilon = epsilon
    self.decay = K.variable(decay, name='decay')
    self.initial_decay = decay

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))

    t = self.iterations + 1
    lr_t = lr / (1. - K.pow(self.beta_1, t))

    shapes = [K.int_shape(p) for p in params]
    # zero init of 1st moment
    ms = [K.zeros(shape) for shape in shapes]
    # zero init of exponentially weighted infinity norm
    us = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + ms + us

    for p, g, m, u in zip(params, grads, ms, us):

      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      u_t = K.maximum(self.beta_2 * u, K.abs(g))
      p_t = p - lr_t * m_t / (u_t + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(u, u_t))

      new_p = p_t
      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)
      self.updates.append(K.update(p, new_p))
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


class Nadam(Optimizer):
  # pylint: disable=line-too-long
  """Nesterov Adam optimizer.

  Much like Adam is essentially RMSprop with momentum,
  Nadam is Adam RMSprop with Nesterov momentum.

  Default parameters follow those provided in the paper.
  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor.

  References:
      - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
      - [On the importance of initialization and momentum in deep
        learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
  """

  # pylint: enable=line-too-long

  def __init__(self,
               lr=0.002,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               schedule_decay=0.004,
               **kwargs):
    super(Nadam, self).__init__(**kwargs)
    self.iterations = K.variable(0., name='iterations')
    self.m_schedule = K.variable(1., name='m_schedule')
    self.lr = K.variable(lr, name='lr')
    self.beta_1 = K.variable(beta_1, name='beta_1')
    self.beta_2 = K.variable(beta_2, name='beta_2')
    self.epsilon = epsilon
    self.schedule_decay = schedule_decay

  def get_updates(self, params, constraints, loss):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    t = self.iterations + 1

    # Due to the recommendations in [2], i.e. warming momentum schedule
    momentum_cache_t = self.beta_1 * (1. - 0.5 *
                                      (K.pow(0.96, t * self.schedule_decay)))
    momentum_cache_t_1 = self.beta_1 * (
        1. - 0.5 * (K.pow(0.96, (t + 1) * self.schedule_decay)))
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
      v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
      v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
      m_t_bar = (1. - momentum_cache_t
                ) * g_prime + momentum_cache_t_1 * m_t_prime

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))

      p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
      new_p = p_t

      # apply constraints
      if p in constraints:
        c = constraints[p]
        new_p = c(new_p)
      self.updates.append(K.update(p, new_p))
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


class TFOptimizer(Optimizer):
  """Wrapper class for native TensorFlow optimizers.
  """

  def __init__(self, optimizer):  # pylint: disable=super-init-not-called
    self.optimizer = optimizer
    self.iterations = K.variable(0., name='iterations')
    self.updates = []

  def get_updates(self, params, constraints, loss):
    if constraints:
      raise ValueError('TF optimizers do not support '
                       'weights constraints. Either remove '
                       'all weights constraints in your model, '
                       'or use a Keras optimizer.')
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

# pylint: disable=invalid-name
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax
nadam = Nadam

# pylint: enable=invalid-name


def serialize(optimizer):
  return serialize_keras_object(optimizer)


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
