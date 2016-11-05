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

"""Module includes reference datasets and utilities to load datasets. It also 
includes methods to generate synthetic data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.base import Dataset

def circles(n_samples=100, noise=None, seed=None, factor=0.8, n_classes=2, *args, **kwargs):
  """Create circles separated by some value

  Args:
    n_samples: int, number of datapoints to generate
    noise: float or None, standard deviation of the Gaussian noise added
    seed: int or None, seed for the noise
    factor: float, size factor of the inner circles with respect to the outer ones
    n_classes: int, number of classes to generate

  Returns:
    Shuffled features and labels for 'circles' synthetic dataset of type `base.Dataset`

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
  linspace = np.linspace(0, 2*np.pi, n_samples // n_classes)
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
    y = np.append(y, label*np.ones(n_samples // n_classes, dtype=np.int32))
  
  # Add more points if n_samples is not divisible by n_classes (unbalanced!)
  extras = n_samples % n_classes
  circ_x = np.append(circ_x, np.cos(np.random.rand(extras)*2*np.pi))
  circ_y = np.append(circ_y, np.sin(np.random.rand(extras)*2*np.pi))
  y = np.append(y, np.zeros(extras, dtype=np.int32))
  
  # Reshape the features/labels
  X = np.vstack((circ_x, circ_y)).T
  y = np.hstack(y)
  
  # Shuffle the data
  indices = np.random.permutation(range(n_samples))
  if noise is not None:
    X += np.random.normal(scale=noise, size=X.shape)
  return Dataset(data=X[indices], target=y[indices])


