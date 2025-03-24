# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_lib", "make_rpaths")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

litert_lib(
    name = "adapter",
    srcs = ["adapter.cc"],
    hdrs = ["adapter.h"],
    data = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/darwinn/compiler:compiler_api_wrapper",
        # copybara:uncomment_end
    ],
    linkopts = [
        # TODO(abhirs): Make this work for OS.
        #copybara:comment_begin(google-only)
        make_rpaths(["platforms/darwinn/compiler"]),
        # copybara:uncomment_end
    ],
    tags = [
        # Don't build/test in OS until libcompiler_api_wrapper.so is available.
        "nobuilder",
        "no_oss",
    ],
    ungrte = False,
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core:dynamic_loading",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "adapter_test",
    srcs = ["adapter_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    linkstatic = 1,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        #Don't build/test in OS until libcompiler_api_wrapper.so is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that uses dlopen.
        "nosan",
        "manual",
    ],
    # This test can only be run on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":adapter",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
