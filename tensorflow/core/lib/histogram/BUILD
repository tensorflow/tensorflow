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

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/lib/monitoring:sampler uses histogram
        "//tensorflow/core/lib/monitoring:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "histogram",
    hdrs = ["histogram.h"],
    deps = [
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "@xla//xla/tsl/lib/histogram",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "@xla//xla/tsl/lib/histogram:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "histogram.h",
        "@xla//xla/tsl/lib/histogram:mobile_hdrs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_histogram_all_headers",
    srcs = [
        "histogram.h",
        "@xla//xla/tsl/lib/histogram:legacy_lib_histogram_all_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
