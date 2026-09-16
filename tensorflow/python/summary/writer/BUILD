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
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "fake_summary_writer",
    testonly = 1,
    srcs = ["fake_summary_writer.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        ":writer_cache",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_library(
    name = "writer",
    srcs = ["writer.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":event_file_writer",
        ":event_file_writer_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:plugin_asset",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "event_file_writer",
    srcs = ["event_file_writer.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
    ],
)

py_library(
    name = "writer_cache",
    srcs = ["writer_cache.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "event_file_writer_v2",
    srcs = ["event_file_writer_v2.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "writer_test",
    size = "small",
    srcs = [
        "writer_test.py",
    ],
    deps = [
        ":writer",
        ":writer_cache",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/summary:plugin_asset",
        "//tensorflow/python/summary:summary_iterator",
        "//tensorflow/python/util:compat",
    ],
)
