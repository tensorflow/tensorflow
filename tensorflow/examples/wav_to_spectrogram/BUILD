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
#   TensorFlow C++ inference example for labeling images.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "wav_to_spectrogram_lib",
    srcs = ["wav_to_spectrogram.cc"],
    hdrs = ["wav_to_spectrogram.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_binary(
    name = "wav_to_spectrogram",
    srcs = ["main.cc"],
    deps = [
        ":wav_to_spectrogram_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "wav_to_spectrogram_test",
    size = "medium",
    srcs = ["wav_to_spectrogram_test.cc"],
    deps = [
        ":wav_to_spectrogram_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
