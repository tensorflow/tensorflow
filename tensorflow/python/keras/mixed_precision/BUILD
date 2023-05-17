# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#   Contains the Keras Mixed Precision API (TensorFlow version).

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # TODO(scottzhu): Remove these two deps and convert the test to integration test.
        "//tensorflow/python/distribute:__pkg__",  # For collective_all_reduce_strategy_test
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "mixed_precision_experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":get_layer_policy",
        ":loss_scale_optimizer",
        ":policy",
    ],
)

py_library(
    name = "policy",
    srcs = [
        "policy.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":device_compatibility_check",
        "//tensorflow/python:framework",
        "//tensorflow/python:mixed_precision_global_state",
    ],
)

py_library(
    name = "device_compatibility_check",
    srcs = ["device_compatibility_check.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:config",
    ],
)

py_library(
    name = "get_layer_policy",
    srcs = ["get_layer_policy.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/keras/engine:base_layer",
    ],
)

py_library(
    name = "autocast_variable",
    srcs = [
        "autocast_variable.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:ps_values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/types:core",
    ],
)

py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:loss_scale",
        "//tensorflow/python/keras/utils:generic_utils",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":loss_scale",
        "//tensorflow/python:cond",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/trackable:base_delegate",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:cond",
        "//tensorflow/python:framework",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras",
    ],
)
