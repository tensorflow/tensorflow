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


def np_array(values, dtype):
  """Creates a NumPy array containing input values.

  It will make a copy of the object.

  In NumPy 2.x and later, strict type casting can lead to errors when values
  overflow the specified dtype. This function addresses this by replacing direct
  np.array(..., dtype=...) calls with np.array(...).astype(...). This allows for
  intended overflows, aligning with the behavior of older NumPy versions.

  Args:
    values: Array_like objects. E.g., a python list, tuple, or an object
    whose __array__ method returns an array.
    dtype: The desired numpy data type for the array.

  Returns:
    A NumPy array with the specified data type.
  """
  if dtype is not None and np.issubdtype(dtype, np.number):
    return np.array(values).astype(dtype)
  else:
    return np.array(values, dtype=dtype)


def np_asarray(values, dtype):
  """Converts input values to a NumPy array.

  It will not make a copy.

  In NumPy 2.x and later, strict type casting can lead to errors when values
  overflow the specified dtype. This function addresses this by replacing direct
  np.array(..., dtype=...) calls with np.array(...).astype(...). This allows for
  intended overflows, aligning with the behavior of older NumPy versions.

  Args:
    values: Array_like objects. E.g., a python list, tuple, or an object
    whose __array__ method returns an array.
    dtype: The desired numpy data type for the array.

  Returns:
    A NumPy array with the specified data type.
  """
  if dtype is not None and np.issubdtype(dtype, np.number):
    return np.asarray(values).astype(dtype)
  else:
    return np.asarray(values, dtype=dtype)
