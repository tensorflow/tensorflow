# Copyright 2025 Google LLC.
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

load("//tensorflow/lite/experimental/litert/runtime/gpu:build_defs.bzl", "litert_angle_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

config_setting(
    name = "angle_on_linux-google",
    constraint_values = [
        "@platforms//os:linux",
    ],
    flag_values = {
        "//third_party/angle:use_angle": "True",
    },
)

cc_library(
    name = "angle",
    deps = [
        "//third_party/GL:EGL_headers",
        "//third_party/GL:GLES2_headers",
        "//third_party/GL:GLES3_headers",
    ] + select({
        # For architectures not supported by ANGLE, use a stub library that
        # allows binaries to link successfully, but aborts the program when any
        # GL function is called.
        "//tools/cc_target_os:platform_ios": [
            ":gl-stub",
        ],
        "//conditions:default": [
            "//third_party/angle:EGL",
            "//third_party/angle:GLESv2",
        ],
    }),
)

cc_test(
    name = "angle_test",
    srcs = ["angle_test.cc"],
    tags = [
        "angle",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":angle",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "@com_google_absl//absl/log:absl_check",
    ] + litert_angle_heapcheck_deps(),
)
