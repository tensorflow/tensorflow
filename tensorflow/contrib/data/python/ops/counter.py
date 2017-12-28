# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""The Counter Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import scan_ops

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


def Counter(start=0, step=1, dtype=dtypes.int64):
  """Creates a `Dataset` of a `step`-separated count startin from `start`.

  For example:

  ```python
  Dataset.count() == [0, 1, 2, ...)
  Dataset.count(2) == [2, 3, ...)
  Dataset.count(2, 5) == [2, 7, 12, ...)
  Dataset.count(0, -1) == [0, -1, -2, ...)
  Dataset.count(10, -1) == [10, 9, ...)
  ```

  Args:
    start: starting value for count.
    step: step size.
    dtype: counter data type.

  Returns:
    A `Dataset` of scalar elements.
  """
  with ops.name_scope("counter"):
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    step = ops.convert_to_tensor(step, dtype=dtype, name="step")
    return dataset_ops.Dataset.from_tensors(0).repeat(None).apply(
        scan_ops.scan(start, lambda state, _: (state + step, state)))
