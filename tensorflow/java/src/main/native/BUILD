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
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Java API using the TensorFlow C library.

load("//tensorflow:tensorflow.bzl", "tf_copts", "tf_cuda_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/java:__pkg__",
        # TODO(ashankar): Temporary hack for the Java API and
        # //third_party/tensorflow/tools/android/inference_interface:android_tensorflow_inference_jni
        # to co-exist in a single shared library. However, the hope is that
        # //third_party/tensorflow/tools/android/inference_interface:android_tensorflow_jni can be
        # removed once the Java API provides feature parity with it.
        "//tensorflow/tools/android/inference_interface:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "native_srcs",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
)

tf_cuda_library(
    name = "native",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    features = select({
        "//tensorflow:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api",
            "//tensorflow/c/eager:c_api",
            "//tensorflow/core:all_kernels",
            "//tensorflow/core:direct_session",
            "//tensorflow/core:ops",
            "@bazel_tools//tools/jdk:jni",
        ],
    }),
    alwayslink = 1,
)
