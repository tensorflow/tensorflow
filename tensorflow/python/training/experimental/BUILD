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
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    strict_deps = True,
    deps = [
        ":loss_scale",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:smart_cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/training:optimizer",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "loss_scale_optimizer_test",
    size = "small",
    srcs = ["loss_scale_optimizer_test.py"],
    strict_deps = True,
    deps = [
        ":loss_scale",
        ":loss_scale_optimizer",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:gradient_descent",
        "//tensorflow/python/training:momentum",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "loss_scale_test",
    size = "medium",
    srcs = ["loss_scale_test.py"],
    strict_deps = True,
    deps = [
        ":loss_scale",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "mixed_precision_global_state",
    srcs = ["mixed_precision_global_state.py"],
    strict_deps = True,
    deps = ["//tensorflow/python/util:tf_export"],
)

py_library(
    name = "mixed_precision",
    srcs = ["mixed_precision.py"],
    strict_deps = True,
    deps = [
        ":loss_scale_optimizer",
        ":mixed_precision_global_state",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:optimizer",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "mixed_precision_test",
    size = "small",
    srcs = ["mixed_precision_test.py"],
    deps = [
        ":loss_scale_optimizer",
        ":mixed_precision",
        ":mixed_precision_global_state",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:gradient_descent",
        "@absl_py//absl/testing:parameterized",
    ],
)
