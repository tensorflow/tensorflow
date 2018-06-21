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
"""MNIST handwritten digits dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.datasets.mnist.load_data')
def load_data(path='mnist.npz'):
  """Loads the MNIST dataset.

  Arguments:
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  License:
      Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
      which is a derivative work from original NIST datasets.
      MNIST dataset is made available under the terms of the
      [Creative Commons Attribution-Share Alike 3.0 license.](
      https://creativecommons.org/licenses/by-sa/3.0/)
  """
  origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  path = get_file(
      path,
      origin=origin_folder + 'mnist.npz',
      file_hash='8a61469f7ea1b51cbae51d4f78837e45')
  with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    
  return (x_train, y_train), (x_test, y_test)
