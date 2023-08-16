#  Copyright 2023 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test .py file for pybind11 files for SavedModelImpl functions LoadSvaedModel & Run."""
from tensorflow.core.tfrt.saved_model.python import _pywrap_saved_model
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class SavedModelLoadSavedModelRunTest(test.TestCase):

  def test_give_me_a_name(self):
    with context.eager_mode(), ops.device("CPU"):
      inputs = [
          constant_op.constant([0, 1, 2, 3, 4, 5, 6, 7]),
          constant_op.constant([1, 5, 8, 9, 21, 54, 67]),
          constant_op.constant([90, 81, 32, 13, 24, 55, 46, 67]),
      ]
    cpp_tensor = _pywrap_saved_model.RunConvertor(inputs)
    return cpp_tensor


if __name__ == "__main__":
  test.main()
