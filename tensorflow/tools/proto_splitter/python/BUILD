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

# Description:
#   Python library for splitting and joining large protos.
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "saved_model",
    srcs = ["saved_model.py"],
    # NOTE(yibaimeng): To be removed when everything is migrated to `pywrap_saved_model.Save`.
    visibility = [
        "//tensorflow:internal",
        "//waymo/ml/deploy/tensorflow:__pkg__",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/tools/proto_splitter:constants",
        "//tensorflow/tools/proto_splitter:split",
        "//tensorflow/tools/proto_splitter:split_graph_def",
    ],
)

py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    strict_deps = True,
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":saved_model",
        ":test_util",
        #internal proto upb dep
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter:constants",
    ],
)

pytype_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    strict_deps = True,
    deps = [
        #internal proto upb dep
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter/python:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)
