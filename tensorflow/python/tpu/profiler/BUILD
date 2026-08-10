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

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "profiler",
    srcs = ["__init__.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/core/profiler:profiler_analysis_proto_py",
        "//tensorflow/python/util:all_util",
        "@tsl//tsl/profiler/protobuf:trace_events_proto_py",
    ],
)

py_library(
    name = "profiler_analysis_pb2_grpc",
    srcs = ["profiler_analysis_pb2_grpc.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/profiler:profiler_analysis_proto_py"],
)

py_library(
    name = "capture_tpu_profile_lib",
    srcs = [
        "capture_tpu_profile.py",
        "version.py",
    ],
    strict_deps = True,
    deps = [
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:profiler_client",
        "//tensorflow/python/profiler:profiler_v2",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@pypi//packaging",
    ],
)

py_binary(
    name = "capture_tpu_profile_bin",
    srcs = ["capture_tpu_profile.py"],
    main = "capture_tpu_profile.py",
    strict_deps = True,
    deps = [
        ":capture_tpu_profile_lib",
        "@absl_py//absl/flags",
        "@pypi//packaging",
    ],
)
