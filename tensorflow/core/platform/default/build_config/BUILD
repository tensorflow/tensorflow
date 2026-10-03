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

# Description:
# Platform-specific build configurations.

load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "platformlib",
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:protos_all_cc",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
        "@fft2d",
        "@highwayhash//:sip_hash",
        "@zlib",
    ],
)

cc_library(
    name = "tensorflow_platform_specific",
    copts = tf_copts(),
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    deps = [],
)
