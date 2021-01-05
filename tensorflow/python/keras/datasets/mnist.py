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
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.datasets.mnist.load_data')
def load_data(path='mnist.npz'):
  """Loads the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

  This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
  along with a test set of 10,000 images.
  More info can be found at the
  [MNIST homepage](http://yann.lecun.com/exdb/mnist/).


  Args:
      path: path where to cache the dataset locally
          (relative to `~/.keras/datasets`).

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

      **x_train, x_test**: uint8 arrays of grayscale image data with shapes
        (num_samples, 28, 28).

      **y_train, y_test**: uint8 arrays of digit labels (integers in range 0-9)
        with shapes (num_samples,).

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
      file_hash=
      '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
  with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
