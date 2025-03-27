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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "gl_utils",
    testonly = True,
    srcs = ["gl_utils.cc"],
    hdrs = ["gl_utils.h"],
    linkopts = select({
        "//tensorflow:android": [
            "-landroid",
            "-lGLESv2",
            "-lEGL",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
    ],
)

cc_test(
    name = "gl_utils_test",
    srcs = ["gl_utils_test.cc"],
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [
        ":gl_utils",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)
