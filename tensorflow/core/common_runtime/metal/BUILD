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
#
# Metal PluggableDevice backend for Apple Silicon GPUs.
#
# Every target here is target_compatible_with macOS, so wildcard builds on
# other platforms skip the package instead of failing to compile Objective-C++.
# Nothing in the framework depends on these targets unless the build was
# configured with --config=metal (see if_metal() in tensorflow.bzl).

load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# Warnings that fire on Apple SDK headers rather than on our own sources.
METAL_COPTS = [
    "-Wno-shorten-64-to-32",
]

objc_library(
    name = "metal_buffer_registry",
    srcs = ["metal_buffer_registry.mm"],
    hdrs = ["metal_buffer_registry.h"],
    copts = METAL_COPTS,
    sdk_frameworks = ["Metal"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
    ],
)

objc_library(
    name = "metal_stream",
    srcs = ["metal_stream.mm"],
    hdrs = ["metal_stream.h"],
    copts = METAL_COPTS,
    sdk_frameworks = ["Metal"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
    ],
)

objc_library(
    name = "metal_stream_executor",
    srcs = ["metal_stream_executor.mm"],
    hdrs = ["metal_stream_executor.h"],
    copts = METAL_COPTS,
    sdk_frameworks = ["Metal"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_buffer_registry",
        ":metal_stream",
        # Headers only. The TF_Status symbols resolve against the framework
        # this plugin is linked into; depending on the implementation target
        # from //tensorflow/core would invert the core -> c layering.
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
    ],
)

objc_library(
    name = "metal_buffer_registry_test_lib",
    testonly = 1,
    srcs = ["metal_buffer_registry_test.mm"],
    copts = METAL_COPTS,
    sdk_frameworks = ["Metal"],
    target_compatible_with = ["@platforms//os:macos"],
    # The gtest cases live in this library; without alwayslink the linker drops
    # their static registration and the test binary runs zero cases.
    alwayslink = 1,
    deps = [
        ":metal_buffer_registry",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "metal_buffer_registry_test",
    size = "small",
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_buffer_registry_test_lib",
        "//tensorflow/core:test_main",
    ],
)
