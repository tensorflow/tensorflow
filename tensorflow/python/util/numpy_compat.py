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

"""Functions for NumPy 1.x vs. 2.x compatibility."""

import numpy as np


def np_array(values, dtype=None, copy=True, order='K'):
  """Creates a NumPy array containing input values.

  It will make a copy of the object.

  In NumPy 2.x and later, strict type casting can lead to errors when values
  overflow the specified dtype. This function addresses this by replacing direct
  np.array(..., dtype=...) calls with np.array(...).astype(...). This allows for
  intended overflows, aligning with the behavior of older NumPy versions.

  Args:
    values: Array_like objects. E.g., a python list, tuple, or an object whose
      __array__ method returns an array.
    dtype: The desired numpy data type for the array.
    copy: Bool. If True (default), then the array data is copied. If None, a
      copy will only be made if __array__ returns a copy, if obj is a nested
      sequence, or if a copy is needed to satisfy any of the other requirements
      (dtype, order, etc.). Note that any copy of the data is shallow, i.e., for
      arrays with object dtype, the new array will point to the same objects.
      For False it raises a ValueError if a copy cannot be avoided.
    order: {‘K’, ‘A’, ‘C’, ‘F’}.

  Returns:
    A NumPy array with the specified data type.
  """
  if dtype is not None and np.issubdtype(dtype, np.number):
    return np.array(values, copy=copy, order=order).astype(dtype)
  else:
    return np.array(values, dtype=dtype, copy=copy, order=order)


def np_asarray(values, dtype=None, order=None, copy=None):
  """Converts input values to a NumPy array.

  It will not make a copy.

  In NumPy 2.x and later, strict type casting can lead to errors when values
  overflow the specified dtype. This function addresses this by replacing direct
  np.array(..., dtype=...) calls with np.array(...).astype(...). This allows for
  intended overflows, aligning with the behavior of older NumPy versions.

  Args:
    values: Array_like objects. E.g., a python list, tuple, or an object whose
      __array__ method returns an array.
    dtype: The desired numpy data type for the array.
    order: {‘C’, ‘F’, ‘A’, ‘K’}.
    copy: bool. If True, then the object is copied. If None then the object is
      copied only if needed, i.e. if __array__ returns a copy, if obj is a
      nested sequence, or if a copy is needed to satisfy any of the other
      requirements (dtype, order, etc.). For False it raises a ValueError if a
      copy cannot be avoided.

  Returns:
    A NumPy array with the specified data type.
  """
  if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
    if dtype is not None and np.issubdtype(dtype, np.number):
      return np.asarray(values, order=order, copy=copy).astype(dtype)
    else:
      return np.asarray(values, dtype=dtype, order=order, copy=copy)
  else:
    return np.asarray(values, dtype=dtype, order=order)
