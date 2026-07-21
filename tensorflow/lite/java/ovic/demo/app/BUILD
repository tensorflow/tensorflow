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

# Sample app for OVIC benchmarking.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

android_binary(
    name = "ovic_benchmarker_binary",
    srcs = [
        "OvicBenchmarkerActivity.java",
    ],
    assets = [
        "//tensorflow/lite/java/ovic/src/testdata:coco_labels.txt",
        "//tensorflow/lite/java/ovic/src/testdata:labels.txt",
        "//tensorflow/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    assets_dir = "",
    custom_package = "ovic.demo.app",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = [
        ".lite",
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    tags = ["manual"],
    deps = [
        "//tensorflow/lite/java:tensorflowlite",
        "//tensorflow/lite/java/ovic:ovicbenchmarkerlib",
        "//tensorflow/lite/java/ovic:ovicdetectionbenchmarkerlib",
    ],
)
