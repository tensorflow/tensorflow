# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""numpy_compat tests."""


import numpy as np

from tensorflow.python.platform import test
from tensorflow.python.util import numpy_compat


class NumpyCompatCopyBehaviorTest(test.TestCase):

  def test_no_copy_new_vs_old(self):

    # Define old_np_asarray to replicate the old code that used .astype(dtype)
    # WITHOUT passing `copy=copy`.
    def old_np_asarray(values, dtype=None, order=None, copy=None):
      if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
        if dtype is not None and np.issubdtype(dtype, np.number):
          return np.asarray(values, order=order, copy=copy).astype(dtype)
        else:
          return np.asarray(values, dtype=dtype, order=order, copy=copy)
      else:
        return np.asarray(values, dtype=dtype, order=order)

    # Test array
    x = np.array([1, 2, 3], dtype=np.float32)

    # Expect old numpy 2.x code to always copy even when copy=None
    y_old = old_np_asarray(x, dtype=np.float32, copy=None)
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
      self.assertIsNot(
          y_old,
          x,
          msg='Old code did NOT copy, but we expect it to always copy.',
      )

    # Expect new code to reuse the array if copy=None
    y_new = numpy_compat.np_asarray(x, dtype=np.float32, copy=None)
    self.assertIs(
        y_new,
        x,
        msg='New code did copy, but we expect it NOT to copy since copy=None.',
    )


if __name__ == '__main__':
  test.main()
