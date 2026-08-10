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

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "data",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/data/experimental",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:readers",
    ],
)
