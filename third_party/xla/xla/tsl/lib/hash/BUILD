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

load(
    "//xla/tsl:tsl.bzl",
    "if_linux_x86_64",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        # tensorflow/compiler/xla/tsl/lib/io/table_builder.cc uses crc functionality
        "//xla/tsl/lib/io:__pkg__",
        # tensorflow/core/lib/hash aliases hash for now
        "//tensorflow/core/lib/hash:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "crc32c",
    srcs = [
        "crc32c.cc",
    ],
    hdrs = ["crc32c.h"],
    # -msse4.2 enables the use of crc32c compiler builtins.
    copts = tsl_copts() + if_linux_x86_64(["-msse4.2"]),
    deps = [
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/crc:crc32c",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:cord",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "crc32c.cc",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/hash:__pkg__"]),
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "crc32c.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/hash:__pkg__"]),
)

filegroup(
    name = "legacy_lib_hash_all_headers",
    srcs = [
        "crc32c.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/hash:__pkg__"]),
)

tsl_cc_test(
    name = "crc32c_test",
    size = "small",
    srcs = ["crc32c_test.cc"],
    deps = [
        ":crc32c",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/strings:cord",
    ],
)
