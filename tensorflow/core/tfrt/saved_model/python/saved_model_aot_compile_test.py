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
import os

import lingvo.compat as tf

from tensorflow.core.tfrt.saved_model.python import _pywrap_saved_model_aot_compile
from tensorflow.python.platform import test


class SavedModelAotCompileTest(test.TestCase):

  def testVerify_saved_model(self):
    outputpath = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR")
    filepath = "learning/brain/tfrt/cpp_tests/gpu_inference/test_data/translate_converted_placed/"
    _pywrap_saved_model_aot_compile.AotCompileSavedModel(
        filepath, _pywrap_saved_model_aot_compile.AotOptions(), outputpath
    )

    # Verifies that .pbtxt is created correctly in the output directory
    self.assertTrue(tf.io.gfile.exists(outputpath + "/aot_saved_model.pbtxt"))


if __name__ == "__main__":
  test.main()
