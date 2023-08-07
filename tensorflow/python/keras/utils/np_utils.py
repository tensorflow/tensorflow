# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Numpy-related utilities."""

import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
  """Converts a class vector (integers) to binary class matrix.

  E.g. for use with categorical_crossentropy.

  Args:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.

  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.

  Example:

  >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
  >>> a = tf.constant(a, shape=[4, 4])
  >>> print(a)
  tf.Tensor(
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

  >>> b = tf.constant([.9, .04, .03, .03,
  ...                  .3, .45, .15, .13,
  ...                  .04, .01, .94, .05,
  ...                  .12, .21, .5, .17],
  ...                 shape=[4, 4])
  >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
  >>> print(np.around(loss, 5))
  [0.10536 0.82807 0.1011  1.77196]

  >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
  >>> print(np.around(loss, 5))
  [0. 0. 0. 0.]

  Raises:
      Value Error: If input contains string value

  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def normalize(x, axis=-1, order=2):
  """Normalizes a Numpy array.

  Args:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. `order=2` for L2 norm).

  Returns:
      A normalized copy of the array.
  """
  l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
  l2[l2 == 0] = 1
  return x / np.expand_dims(l2, axis)
