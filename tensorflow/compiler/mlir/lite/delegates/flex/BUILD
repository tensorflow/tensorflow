# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load(
    "//tensorflow:tensorflow.bzl",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
    "tf_features_nolayering_check_if_ios",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:special_rules.bzl", "internal_visibility_allowlist")
load("//tensorflow/compiler/mlir/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library")

# LINT.IfChange

default_visibility = [
    "//tensorflow/compiler/mlir/lite:__subpackages__",
    "//tensorflow/lite/android:__subpackages__",
    "//tensorflow/lite/toco/tflite:__subpackages__",
]

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

exports_files([
    "delegate.h",
    "exported_symbols.lds",
    "version_script.lds",
])

cc_library(
    name = "allowlisted_flex_ops_lib",
    srcs = [
        "allowlisted_flex_ops.cc",
    ],
    hdrs = [
        "allowlisted_flex_ops.h",
        "allowlisted_flex_ops_internal.h",
    ],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    visibility = internal_visibility_allowlist(),
    deps = if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
    ]),
)

tf_cc_test(
    name = "allowlisted_flex_ops_test",
    size = "small",
    srcs = [
        "allowlisted_flex_ops_test.cc",
    ],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":allowlisted_flex_ops_lib",
        ":delegate",
        "@com_google_googletest//:gtest_main",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
    ]),
)

tflite_flex_cc_library(
    name = "delegate",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

# LINT.ThenChange(//tensorflow/compiler/mlir/lite/delegates/flex/BUILD)
