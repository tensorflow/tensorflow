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
#   Benchmark utility that can run on desktop and Android.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "benchmark_model_lib",
    testonly = 1,
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = [
        "benchmark_model.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
            "//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:framework_lite",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensorflow",
            "//tensorflow/core:test",
        ],
    }) + [
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/util:stats_calculator_portable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "benchmark_model_test",
    size = "medium",
    srcs = ["benchmark_model_test.cc"],
    deps = [
        ":benchmark_model_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# This binary may be built for either desktop or Android.
# A typical Android build command will look like the following:
# bazel build tensorflow/core:portable_tensorflow_lib \
# --crosstool_top=//external:android/crosstool \
# --cpu=armeabi-v7a \
# --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
# --config monolithic
tf_cc_binary(
    name = "benchmark_model",
    testonly = 1,
    srcs = ["benchmark_model_main.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-s",
            "-landroid",
            "-latomic",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":benchmark_model_lib"],
)
