# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-bad-import-order

"""Data augmentation that will be used in data_load.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np


def time_wrapping(molecule, denominator, data):
  """Generate (molecule/denominator)x speed data."""
  tmp_data = [[0
               for i in range(len(data[0]))]
              for j in range((int(len(data) / molecule) - 1) * denominator)]
  for i in range(int(len(data) / molecule) - 1):
    for j in range(len(data[i])):
      for k in range(denominator):
        tmp_data[denominator * i +
                 k][j] = (data[molecule * i + k][j] * (denominator - k) +
                          data[molecule * i + k + 1][j] * k) / denominator
  return tmp_data


def augment_data(original_data, original_label):
  """Perform data augmentation."""
  new_data = []
  new_label = []
  for idx, (data, label) in enumerate(zip(original_data, original_label)):  # pylint: disable=unused-variable
    # Original data
    new_data.append(data)
    new_label.append(label)
    # Sequence shift
    for num in range(5):  # pylint: disable=unused-variable
      new_data.append((np.array(data, dtype=np.float32) +
                       (random.random() - 0.5) * 200).tolist())
      new_label.append(label)
    # Random noise
    tmp_data = [[0 for i in range(len(data[0]))] for j in range(len(data))]
    for num in range(5):
      for i in range(len(tmp_data)):
        for j in range(len(tmp_data[i])):
          tmp_data[i][j] = data[i][j] + 5 * random.random()
      new_data.append(tmp_data)
      new_label.append(label)
    # Time warping
    fractions = [(3, 2), (5, 3), (2, 3), (3, 4), (9, 5), (6, 5), (4, 5)]
    for molecule, denominator in fractions:
      new_data.append(time_wrapping(molecule, denominator, data))
      new_label.append(label)
    # Movement amplification
    for molecule, denominator in fractions:
      new_data.append(
          (np.array(data, dtype=np.float32) * molecule / denominator).tolist())
      new_label.append(label)
  return new_data, new_label
