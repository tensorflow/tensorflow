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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "stacktrace",
    linkopts = ["-labsl_stacktrace"],
    deps = [
        ":debugging_internal",
    ],
)

cc_library(
    name = "symbolize",
    linkopts = ["-labsl_symbolize"],
    deps = [
        ":debugging_internal",
        ":demangle_internal",
        "//absl/base",
        "//absl/base:dynamic_annotations",
        "//absl/base:malloc_internal",
        "//absl/base:raw_logging_internal",
        "//absl/strings",
    ],
)

cc_library(
    name = "failure_signal_handler",
    linkopts = [
        "-labsl_failure_signal_handler",
        "-labsl_examine_stack",
    ],
    deps = [
        ":stacktrace",
        ":symbolize",
        "//absl/base",
        "//absl/base:errno_saver",
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "debugging_internal",
    linkopts = ["-labsl_debugging_internal"],
    deps = [
        "//absl/base:dynamic_annotations",
        "//absl/base:errno_saver",
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "demangle_internal",
    linkopts = ["-labsl_demangle_internal"],
    deps = [
        "//absl/base",
    ],
)

cc_library(
    name = "leak_check",
    linkopts = ["-labsl_leak_check"],
)

cc_library(
    name = "leak_check_disable",
    linkopts = ["-labsl_leak_check_disable"],
    alwayslink = 1,
)
