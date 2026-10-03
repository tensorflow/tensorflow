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

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

py_test(
    name = "public_symbol_test",
    srcs = ["public_symbol_test.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "np_config_test",
    srcs = ["np_config_test.py"],
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/ops/numpy_ops:np_config",
    ],
)
