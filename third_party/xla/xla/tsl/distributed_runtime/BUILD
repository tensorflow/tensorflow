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

# Description:
# Distributed runtime modules for machine learning, which allows coordination between multiple
# processes for distributed operations.

load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "call_options",
    srcs = ["call_options.cc"],
    hdrs = ["call_options.h"],
    deps = [
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "call_options.h",
    ],
)
