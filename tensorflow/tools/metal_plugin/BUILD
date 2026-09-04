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
# A PluggableDevice for Apple silicon, built with Bazel.
#
# The Makefile beside this file builds the same sources against an installed
# TensorFlow wheel, which is how the plugin ships on PyPI and the only way to
# build it on a machine that is not set up for a TensorFlow Bazel build. The
# two agree on the sources and differ in what they compile against: the
# Makefile turns on TF_METAL_OUT_OF_TREE and TF_CAPI_WEAK, which are no-ops in
# the tree and are therefore absent here.
#
# Nothing in TensorFlow's CI can build these targets: there is no macOS runner
# in .github/workflows, and the tests want a Metal device besides. That is the
# same position as //tensorflow/lite/delegates/gpu/metal, which has carried
# sixty Objective-C++ sources under objc_library on those terms.

load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

# Apple silicon only. Every source here is Objective-C++ against Metal and
# MetalPerformanceShadersGraph, so on another platform there is nothing to
# build rather than something that builds and then cannot run.
MACOS_ONLY = select({
    "//tensorflow:macos": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

# The backend keeps the TensorFlow paths it would have in the tree
# (tensorflow/core/common_runtime/metal/...), so that moving a file either way
# is a copy rather than a patch. That is why the include root is src and not
# the package.
objc_library(
    name = "metal_backend",
    hdrs = glob([
        "src/tensorflow/core/common_runtime/metal/*.h",
        "src/tensorflow/core/common_runtime/metal/kernels/*.h",
    ]),
    includes = ["src"],
    # non_arc_srcs rather than srcs: the backend manages its Metal objects with
    # explicit retain and release, and objc_library compiles srcs with
    # automatic reference counting, which rejects both outright.
    non_arc_srcs = glob([
        "src/tensorflow/core/common_runtime/metal/*.mm",
        "src/tensorflow/core/common_runtime/metal/kernels/*.mm",
    ]),
    sdk_frameworks = [
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ],
    tags = ["manual"],
    target_compatible_with = MACOS_ONLY,
    # The kernels register themselves from static initialisers and nothing in
    # plugin_init.cc names a symbol of theirs, so without this the linker drops
    # every object in the library and the plugin loads with no kernels in it.
    alwayslink = 1,
    deps = [
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_datatype_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_tensor_hdrs",
        "//tensorflow/c/experimental/grappler:grappler_hdrs",
        "//tensorflow/c/experimental/pluggable_profiler:pluggable_profiler_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/core/platform:fingerprint",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
    ],
)

# The entry points TensorFlow looks up by name after dlopen. Built the way
# //tensorflow/c/experimental/stream_executor/test builds its own
# PluggableDevice: a shared object over the C API headers, with no TensorFlow
# implementation linked into it.
tf_cc_shared_object(
    name = "metal_plugin.so",
    srcs = ["src/plugin_init.cc"],
    includes = ["src"],
    tags = ["manual"],
    target_compatible_with = MACOS_ONLY,
    visibility = ["//visibility:public"],
    deps = [
        ":metal_backend",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c/experimental/grappler:grappler_hdrs",
        "//tensorflow/c/experimental/pluggable_profiler:pluggable_profiler_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
    ],
)

# Takes the built plugin as its argument and drives the StreamExecutor C API
# against a real device. Tagged manual because no test runner here has a Metal
# device, not because it is expected to fail.
objc_library(
    name = "stream_executor_test_lib",
    testonly = 1,
    includes = ["src"],
    non_arc_srcs = ["tests/stream_executor_test.mm"],
    sdk_frameworks = ["Foundation"],
    tags = ["manual"],
    target_compatible_with = MACOS_ONLY,
    deps = [
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
    ],
)

cc_test(
    name = "stream_executor_test",
    args = ["$(location :metal_plugin.so)"],
    data = [":metal_plugin.so"],
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    target_compatible_with = MACOS_ONLY,
    deps = [":stream_executor_test_lib"],
)
