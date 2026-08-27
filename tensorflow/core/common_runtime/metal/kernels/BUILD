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
# Op kernels for the Metal PluggableDevice backend, registered through the
# TensorFlow Kernel C API. As in the parent package, every target is macOS-only
# and is only reachable from a build configured with --config=metal.

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/common_runtime/metal:__subpackages__",
    ],
    licenses = ["notice"],
)

METAL_COPTS = [
    "-Wno-shorten-64-to-32",
]

objc_library(
    name = "metal_kernel_util",
    srcs = ["metal_kernel_util.mm"],
    hdrs = ["metal_kernel_util.h"],
    copts = METAL_COPTS,
    sdk_frameworks = ["Metal"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_tensor_hdrs",
        "//tensorflow/core/common_runtime/metal:metal_buffer_registry",
        "//tensorflow/core/common_runtime/metal:metal_stream",
    ],
)

objc_library(
    name = "metal_shader_library",
    srcs = ["metal_shader_library.mm"],
    hdrs = ["metal_shader_library.h"],
    copts = METAL_COPTS,
    sdk_frameworks = ["Metal"],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core/common_runtime/metal:metal_stream",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
    ],
)

objc_library(
    name = "metal_kernels",
    srcs = [
        "metal_elementwise_ops.mm",
        "metal_identity_op.mm",
        "metal_kernels.mm",
        "metal_matmul_op.mm",
    ],
    hdrs = ["metal_kernels.h"],
    copts = METAL_COPTS,
    sdk_frameworks = [
        "Metal",
        "MetalPerformanceShaders",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":metal_kernel_util",
        ":metal_shader_library",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_datatype_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_tensor_hdrs",
        "//tensorflow/core/common_runtime/metal:metal_platform",
        "//tensorflow/core/common_runtime/metal:metal_stream",
        "@com_google_absl//absl/log",
    ],
)
