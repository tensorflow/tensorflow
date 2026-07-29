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
        # tensorflow/core/lib/gtl uses hash functions as part of their flatset/map representation
        "//tensorflow/core/lib/gtl:__pkg__",
        # tensorflow/core/lib/io/table_builder.cc uses crc functionality
        "//tensorflow/core/lib/io:__pkg__",
        # tensorflow/core/framework:tensor depends on hash
        "//tensorflow/core/framework:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "crc32c",
    hdrs = ["crc32c.h"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:types",
        "@xla//xla/tsl/lib/hash:crc32c",
    ],
)

cc_library(
    name = "hash",
    hdrs = ["hash.h"],
    deps = [
        "//tensorflow/core/platform:hash",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_hdrs_no_runtime",
    srcs = [
        "hash.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "@xla//xla/tsl/lib/hash:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "crc32c.h",
        "@xla//xla/tsl/lib/hash:mobile_hdrs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_hash_all_headers",
    srcs = [
        "crc32c.h",
        "hash.h",
        "@xla//xla/tsl/lib/hash:legacy_lib_hash_all_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "hash.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
