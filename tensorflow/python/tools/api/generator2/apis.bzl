# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""generate_api API definitions."""

load(":patterns.bzl", "compile_patterns")

APIS = {
    "tf_keras": {
        "decorator": "tensorflow.python.util.tf_export.keras_export",
        "target_patterns": compile_patterns([
            "//third_party/py/tf_keras/...",
        ]),
    },
    "tensorflow": {
        "decorator": "tensorflow.python.util.tf_export.tf_export",
        "target_patterns": compile_patterns([
            "//tensorflow/python/...",
            "//tensorflow/dtensor/python:accelerator_util",
            "//tensorflow/dtensor/python:api",
            "//tensorflow/dtensor/python:config",
            "//tensorflow/dtensor/python:d_checkpoint",
            "//tensorflow/dtensor/python:d_variable",
            "//tensorflow/dtensor/python:input_util",
            "//tensorflow/dtensor/python:layout",
            "//tensorflow/dtensor/python:mesh_util",
            "//tensorflow/dtensor/python:tpu_util",
            "//tensorflow/dtensor/python:save_restore",
            "//tensorflow/lite/python/...",
            "//tensorflow/python:modules_with_exports",
            "//tensorflow/lite/tools/optimize/debugging/python:all",
            "//tensorflow/compiler/mlir/quantization/tensorflow/python:all",
        ]),
    },
}
