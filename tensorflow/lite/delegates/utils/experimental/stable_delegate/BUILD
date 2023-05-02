# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Utils package for the stable delegate APIs.
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object", "tflite_linkopts_no_undefined")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

cc_library_with_tflite(
    name = "delegate_loader",
    srcs = ["delegate_loader.cc"],
    hdrs = ["delegate_loader.h"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    tflite_deps = ["//tensorflow/lite/experimental/acceleration/configuration/c:stable_delegate"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "//tensorflow/lite/tools:logging",
    ],
)

cc_library_with_tflite(
    name = "stable_delegate_interface",
    hdrs = ["stable_delegate_interface.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = ["//tensorflow/lite/experimental/acceleration/configuration/c:stable_delegate"],
    visibility = ["//visibility:public"],
)

tflite_cc_shared_object(
    name = "tensorflowlite_stable_xnnpack_delegate",
    testonly = True,
    srcs = ["stable_xnnpack_delegate.cc"],
    defines = ["TFL_STABLE_DELEGATE_COMPILE_LIBRARY"],
    linkopts = tflite_linkopts_no_undefined() + select({
        "//tensorflow:windows": [],
        # iOS linker doesn't support -version-script.
        "//tensorflow:ios": [
            # TODO(b/260213307): support hiding unnecessary symbols on iOS.
        ],
        "//conditions:default": [
            # Expose necessary symbols only.
            "-Wl,--version-script,$(location //tensorflow/lite/delegates/utils/experimental/stable_delegate:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    visibility = ["//visibility:public"],
    deps = [
        ":stable_delegate_interface",
        ":version_script.lds",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/experimental/acceleration/configuration/c:stable_delegate",
        "//tensorflow/lite/experimental/acceleration/configuration/c:xnnpack_plugin",
    ],
)

cc_test(
    name = "delegate_loader_test",
    srcs = ["delegate_loader_test.cc"],
    data = ["//tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate"],
    deps = [
        ":delegate_loader",
        "//tensorflow/lite/delegates/utils/experimental/sample_stable_delegate",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_settings_json_parser",
    srcs = [
        "tflite_settings_json_parser.cc",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs_contents-inl.h",
    ],
    hdrs = ["tflite_settings_json_parser.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/tools:logging",
        "@flatbuffers",
    ],
)

build_test(
    name = "stable_delegate_build_test",
    targets = [
        ":delegate_loader",
        ":test_xnnpack_settings.json",
        ":tensorflowlite_stable_xnnpack_delegate",
        ":tflite_settings_json_parser",
    ],
)

cc_test(
    name = "tflite_settings_json_parser_test",
    srcs = ["tflite_settings_json_parser_test.cc"],
    data = [
        ":test_xnnpack_settings.json",
        "//tensorflow/lite/tools/delegates/experimental/stable_delegate:test_invalid_settings.json",
    ],
    deps = [
        ":tflite_settings_json_parser",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

exports_files(
    srcs = [
        "test_xnnpack_settings.json",
        "version_script.lds",
    ],
    visibility = ["//visibility:public"],
)
