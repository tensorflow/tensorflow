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

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

android_binary(
    name = "TfLiteCameraDemo",
    srcs = glob(["java/**/*.java"]),
    assets = [
        "//tensorflow/lite/java/demo/app/src/main/assets:labels_mobilenet_quant_v1_224.txt",
        "@tflite_mobilenet_quant//:mobilenet_v1_1.0_224_quant.tflite",
        "@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite",
    ],
    assets_dir = "",
    custom_package = "com.example.android.tflitecamerademo",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = [
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        "//tensorflow/lite/java:tensorflowlite",
        "//tensorflow/lite/java:tensorflowlite_gpu",
        "//tensorflow/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
    ],
)
