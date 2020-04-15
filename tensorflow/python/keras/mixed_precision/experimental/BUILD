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

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = [
        # TODO(scottzhu): Remove these two deps and convert the test to integration test.
        "//tensorflow/python:__pkg__",  # For loss_scale_optimizer_test
        "//tensorflow/python/distribute:__pkg__",  # For collective_all_reduce_strategy_test
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "mixed_precision_experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":device_compatibility_check",
        "//tensorflow/python:framework",
        "//tensorflow/python:mixed_precision_global_state",
    ],
)

py_test(
    name = "policy_test",
    size = "medium",
    srcs = [
        "policy_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":policy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras/mixed_precision/experimental:loss_scale_optimizer",
        "//tensorflow/python/keras/optimizer_v2",
    ],
)

py_library(
    name = "device_compatibility_check",
    srcs = ["device_compatibility_check.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:device_lib",
        "//tensorflow/python:gpu_util",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "device_compatibility_check_test",
    srcs = ["device_compatibility_check_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":device_compatibility_check",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras:combinations",
    ],
)

py_library(
    name = "get_layer_policy",
    srcs = ["get_layer_policy.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/keras/engine:base_layer",
    ],
)

py_test(
    name = "get_layer_policy_test",
    srcs = ["get_layer_policy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":get_layer_policy",
        ":policy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras/layers",
    ],
)

py_library(
    name = "autocast_variable",
    srcs = [
        "autocast_variable.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "autocast_variable_test",
    size = "medium",
    srcs = ["autocast_variable_test.py"],
    python_version = "PY3",
    deps = [
        ":autocast_variable",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras/optimizer_v2",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:loss_scale",
        "//tensorflow/python/keras/utils:generic_utils",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss_scale",
        "//tensorflow/python:loss_scale",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/keras/optimizer_v2",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "loss_scale_optimizer_test",
    size = "small",
    srcs = ["loss_scale_optimizer_test.py"],
    python_version = "PY3",
    deps = [
        ":loss_scale_optimizer",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python/distribute:central_storage_strategy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
    ],
)

cuda_py_test(
    name = "loss_scale_benchmark",
    size = "medium",
    srcs = ["loss_scale_benchmark.py"],
    deps = [
        ":loss_scale_optimizer",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:loss_scaling_gradient_tape",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/keras",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python/keras",
    ],
)

cuda_py_test(
    name = "keras_test",
    size = "medium",
    srcs = ["keras_test.py"],
    data = [
        "//tensorflow/python/keras/mixed_precision/experimental/testdata:lso_ckpt_tf2.2",
        "//tensorflow/python/keras/mixed_precision/experimental/testdata:lso_savedmodel_tf2.2",
    ],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_pip",
        "no_windows",  # b/139083295: bfloat16 tests fail on Windows
    ],
    deps = [
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:central_storage_strategy",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "layer_correctness_test",
    size = "medium",
    srcs = ["layer_correctness_test.py"],
    python_version = "PY3",
    shard_count = 10,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)
