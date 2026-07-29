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
# Internal helper function to test TF Lite API.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_java//java:defs.bzl", "java_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

android_library(
    name = "testhelper",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//tensorflow/lite/java:tensorflowlite_java",
    ],
)

java_library(
    name = "testhelper_javalib",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//tensorflow/lite/java:tensorflowlite_javalib",
    ],
)
