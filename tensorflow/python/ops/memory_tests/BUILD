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

# python/ops/memory_tests package

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "custom_gradient_memory_test",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    xla_enable_strict_auto_jit = False,  # XLA are enabled explicitly in XLA memory tests.
    deps = [
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
        "@xla//xla/service:hlo_proto_py",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_strict_test(
    name = "custom_gradient_memory_test_tpu",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    main = "custom_gradient_memory_test.py",
    deps = [
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
        "@xla//xla/service:hlo_proto_py",
    ] + tf_additional_xla_deps_py(),
)
