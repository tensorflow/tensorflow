# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Description:
#   Keras saving and loading files for SavedModel.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/python/keras/saving:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "load_context",
    srcs = ["load_context.py"],
    visibility = ["//tensorflow/python/distribute:__pkg__"],
)

py_library(
    name = "saved_model",
    srcs = [
        "base_serialization.py",
        "constants.py",
        "json_utils.py",
        "layer_serialization.py",
        "load.py",
        "load_context.py",
        "metric_serialization.py",
        "model_serialization.py",
        "network_serialization.py",
        "save.py",
        "save_impl.py",
        "serialized_attributes.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
    ],
)
