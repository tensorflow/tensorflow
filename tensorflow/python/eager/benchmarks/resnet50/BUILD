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
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "resnet50",
    srcs = ["resnet50.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "resnet50_test_util",
    srcs = ["resnet50_test_util.py"],
    strict_deps = True,
    deps = ["//tensorflow:tensorflow_py_no_contrib"],
)

py_library(
    name = "resnet50_test_lib",
    srcs = ["resnet50_test.py"],
    strict_deps = True,
    deps = [
        ":resnet50",
        ":resnet50_test_util",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:test_lib",
    ],
)

cuda_py_strict_test(
    name = "resnet50_test",
    size = "medium",
    srcs = ["resnet50_test.py"],
    shard_count = 4,
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/141617449): needs investigation
        "optonly",
        "oss_serial",
        "v1only",
    ],
    deps = [
        ":resnet50",
        ":resnet50_test_util",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:test_lib",
    ],
)

cuda_py_strict_test(
    name = "hvp_test",
    size = "medium",
    srcs = ["hvp_test.py"],
    shard_count = 7,
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/141617449): needs investigation
        "optonly",
        "oss_serial",
        "v1only",
    ],
    # Times out
    xla_enable_strict_auto_jit = False,
    deps = [
        ":resnet50",
        ":resnet50_test_util",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/eager:forwardprop",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "resnet50_graph_test",
    size = "medium",
    srcs = ["resnet50_graph_test.py"],
    shard_count = 4,
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/141617449): needs investigation
        "optonly",
        "oss_serial",
    ],
    deps = [
        ":resnet50",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
    ],
)
