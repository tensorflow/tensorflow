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

# Description: OSS only cluster resolvers

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["tpu_cluster_resolver.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//tensorflow/python/tpu:tpu_system_metadata",
        "//tensorflow/python/tpu/client",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
    ] + tf_additional_rpc_deps(),
)

tf_py_strict_test(
    name = "tpu_cluster_resolver_py_test",
    size = "small",
    srcs = ["tpu_cluster_resolver_test.py"],
    grpc_enabled = True,
    main = "tpu_cluster_resolver_test.py",
    deps = [
        ":tpu_cluster_resolver_py",
        "//tensorflow/core/protobuf/tpu:topology_proto_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu/client",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:compat",
    ],
)
