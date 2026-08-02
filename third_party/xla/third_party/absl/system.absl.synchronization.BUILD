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

# Internal data structure for efficiently detecting mutex dependency cycles
cc_library(
    name = "graphcycles_internal",
    linkopts = ["-labsl_graphcycles_internal"],
    visibility = [
        "//absl:__subpackages__",
    ],
    deps = [
        "//absl/base",
        "//absl/base:malloc_internal",
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "synchronization",
    linkopts = [
        "-labsl_synchronization",
        "-pthread",
    ],
    deps = [
        ":graphcycles_internal",
        "//absl/base",
        "//absl/base:atomic_hook",
        "//absl/base:dynamic_annotations",
        "//absl/base:malloc_internal",
        "//absl/base:raw_logging_internal",
        "//absl/debugging:stacktrace",
        "//absl/debugging:symbolize",
        "//absl/time",
    ],
)
