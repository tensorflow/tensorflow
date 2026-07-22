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
load("//tensorflow/core/platform:distribute.bzl", "distribute_py_strict_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

distribute_py_strict_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    exec_properties = select({
        # copybara:uncomment_begin(google-only)
        # "@bazel_tools//tools/cpp:asan_build": {"mem": "16g"},
        # copybara:uncomment_end
        "//conditions:default": None,
    }),
    tags = [
        "no_windows",  # TODO(b/171350360)
        "nomultivm",  # multi_worker_test_base incompatible with multivm base
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:lookup_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mwms_peer_failure_test",
    size = "medium",
    srcs = ["mwms_peer_failure_test.py"],
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/227372713)
        "notsan",  # b/195248428
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "mwms_peer_failure_test_lib",
    srcs = ["mwms_peer_failure_test.py"],
    strict_deps = True,
    visibility = ["//learning/brain/runtime/python:__pkg__"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/eager:test",
    ],
)

tpu_py_strict_test(
    name = "tpu_memory_test",
    size = "medium",
    srcs = ["tpu_memory_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = True,
    disable_tfrt = False,
    disable_v2 = True,
    tags = ["no_oss"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:flags",
    ],
)
