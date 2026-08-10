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
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/data/experimental/ops:dataset_ops",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/experimental/ops:iterator_model_ops",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/experimental/ops:lookup_ops",
        "//tensorflow/python/data/experimental/ops:parsing_ops",
        "//tensorflow/python/data/experimental/ops:random_ops",
        "//tensorflow/python/data/experimental/service",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util:all_util",
    ],
)
