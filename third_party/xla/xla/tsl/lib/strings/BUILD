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

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "proto_serialization",
    srcs = ["proto_serialization.cc"],
    hdrs = ["proto_serialization.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/lib/gtl:inlined_vector",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/platform:protobuf",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "proto_serialization.cc",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)
