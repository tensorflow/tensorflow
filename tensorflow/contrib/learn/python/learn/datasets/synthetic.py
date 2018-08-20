# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Synthetic dataset generators (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.base import Dataset
from tensorflow.python.util.deprecation import deprecated


@deprecated(None, 'Consider using synthetic datasets from scikits.learn.')
def circles(n_samples=100,
            noise=None,
            seed=None,
            factor=0.8,
            n_classes=2,
            *args,
            **kwargs):
  """Create circles separated by some value

  Args:
    n_samples: int, number of datapoints to generate
    noise: float or None, standard deviation of the Gaussian noise added
    seed: int or None, seed for the noise
    factor: float, size factor of the inner circles with respect to the outer
      ones
    n_classes: int, number of classes to generate

  Returns:
    Shuffled features and labels for 'circles' synthetic dataset of type
    `base.Dataset`

  Note:
    The multi-class support might not work as expected if `noise` is enabled

  TODO:
    - Generation of unbalanced data

  Credit goes to (under BSD 3 clause):
    B. Thirion,
    G. Varoquaux,
    A. Gramfort,
    V. Michel,
    O. Grisel,
    G. Louppe,
    J. Nothman
  """
  if seed is not None:
    np.random.seed(seed)
  # Algo: 1) Generate initial circle, 2) For ever class generate a smaller radius circle
  linspace = np.linspace(0, 2 * np.pi, n_samples // n_classes)
  circ_x = np.empty(0, dtype=np.int32)
  circ_y = np.empty(0, dtype=np.int32)
  base_cos = np.cos(linspace)
  base_sin = np.sin(linspace)

  y = np.empty(0, dtype=np.int32)
  for label in range(n_classes):
    circ_x = np.append(circ_x, base_cos)
    circ_y = np.append(circ_y, base_sin)
    base_cos *= factor
    base_sin *= factor
    y = np.append(y, label * np.ones(n_samples // n_classes, dtype=np.int32))

  # Add more points if n_samples is not divisible by n_classes (unbalanced!)
  extras = n_samples % n_classes
  circ_x = np.append(circ_x, np.cos(np.random.rand(extras) * 2 * np.pi))
  circ_y = np.append(circ_y, np.sin(np.random.rand(extras) * 2 * np.pi))
  y = np.append(y, np.zeros(extras, dtype=np.int32))

  # Reshape the features/labels
  X = np.vstack((circ_x, circ_y)).T
  y = np.hstack(y)

  # Shuffle the data
  indices = np.random.permutation(range(n_samples))
  if noise is not None:
    X += np.random.normal(scale=noise, size=X.shape)
  return Dataset(data=X[indices], target=y[indices])


@deprecated(None, 'Consider using synthetic datasets from scikits.learn.')
def spirals(n_samples=100,
            noise=None,
            seed=None,
            mode='archimedes',
            n_loops=2,
            *args,
            **kwargs):
  """Create spirals

  Currently only binary classification is supported for spiral generation

  Args:
    n_samples: int, number of datapoints to generate
    noise: float or None, standard deviation of the Gaussian noise added
    seed: int or None, seed for the noise
    n_loops: int, number of spiral loops, doesn't play well with 'bernoulli'
    mode: str, how the spiral should be generated. Current implementations:
      'archimedes': a spiral with equal distances between branches
      'bernoulli': logarithmic spiral with branch distances increasing
      'fermat': a spiral with branch distances decreasing (sqrt)

  Returns:
    Shuffled features and labels for 'spirals' synthetic dataset of type
    `base.Dataset`

  Raises:
    ValueError: If the generation `mode` is not valid

  TODO:
    - Generation of unbalanced data
  """
  n_classes = 2  # I am not sure how to make it multiclass

  _modes = {
      'archimedes': _archimedes_spiral,
      'bernoulli': _bernoulli_spiral,
      'fermat': _fermat_spiral
  }

  if mode is None or mode not in _modes:
    raise ValueError('Cannot generate spiral with mode %s' % mode)

  if seed is not None:
    np.random.seed(seed)
  linspace = np.linspace(0, 2 * n_loops * np.pi, n_samples // n_classes)
  spir_x = np.empty(0, dtype=np.int32)
  spir_y = np.empty(0, dtype=np.int32)

  y = np.empty(0, dtype=np.int32)
  for label in range(n_classes):
    base_cos, base_sin = _modes[mode](linspace, label * np.pi, *args, **kwargs)
    spir_x = np.append(spir_x, base_cos)
    spir_y = np.append(spir_y, base_sin)
    y = np.append(y, label * np.ones(n_samples // n_classes, dtype=np.int32))

  # Add more points if n_samples is not divisible by n_classes (unbalanced!)
  extras = n_samples % n_classes
  if extras > 0:
    x_extra, y_extra = _modes[mode](np.random.rand(extras) * 2 * np.pi, *args,
                                    **kwargs)
    spir_x = np.append(spir_x, x_extra)
    spir_y = np.append(spir_y, y_extra)
    y = np.append(y, np.zeros(extras, dtype=np.int32))

  # Reshape the features/labels
  X = np.vstack((spir_x, spir_y)).T
  y = np.hstack(y)

  # Shuffle the data
  indices = np.random.permutation(range(n_samples))
  if noise is not None:
    X += np.random.normal(scale=noise, size=X.shape)
  return Dataset(data=X[indices], target=y[indices])


def _archimedes_spiral(theta, theta_offset=0., *args, **kwargs):
  """Return Archimedes spiral

  Args:
    theta: array-like, angles from polar coordinates to be converted
    theta_offset: float, angle offset in radians (2*pi = 0)
  """
  x, y = theta * np.cos(theta + theta_offset), theta * np.sin(
      theta + theta_offset)
  x_norm = np.max(np.abs(x))
  y_norm = np.max(np.abs(y))
  x, y = x / x_norm, y / y_norm
  return x, y


def _bernoulli_spiral(theta, theta_offset=0., *args, **kwargs):
  """Return Equiangular (Bernoulli's) spiral

  Args:
    theta: array-like, angles from polar coordinates to be converted
    theta_offset: float, angle offset in radians (2*pi = 0)

  Kwargs:
    exp_scale: growth rate of the exponential
  """
  exp_scale = kwargs.pop('exp_scale', 0.1)

  x, y = np.exp(exp_scale * theta) * np.cos(theta + theta_offset), np.exp(
      exp_scale * theta) * np.sin(theta + theta_offset)
  x_norm = np.max(np.abs(x))
  y_norm = np.max(np.abs(y))
  x, y = x / x_norm, y / y_norm
  return x, y


def _fermat_spiral(theta, theta_offset=0., *args, **kwargs):
  """Return Parabolic (Fermat's) spiral

  Args:
    theta: array-like, angles from polar coordinates to be converted
    theta_offset: float, angle offset in radians (2*pi = 0)
  """
  x, y = np.sqrt(theta) * np.cos(theta + theta_offset), np.sqrt(theta) * np.sin(
      theta + theta_offset)
  x_norm = np.max(np.abs(x))
  y_norm = np.max(np.abs(y))
  x, y = x / x_norm, y / y_norm
  return x, y
