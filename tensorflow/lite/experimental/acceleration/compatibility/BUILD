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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_android_library", "flatbuffer_cc_library", "flatbuffer_java_library")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_gpu_tests_tags")
load("//tensorflow/lite:special_rules.bzl", "tflite_extra_gles_deps", "tflite_portable_test_suite")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

flatbuffer_cc_library(
    name = "database_fbs",
    srcs = ["database.fbs"],
)

exports_files(srcs = ["database.fbs"])

flatbuffer_java_library(
    name = "database_fbs_java",
    srcs = ["database.fbs"],
    package_prefix = "org.tensorflow",
)

flatbuffer_android_library(
    name = "database_fbs_android",
    srcs = ["database.fbs"],
    package_prefix = "org.tensorflow",
)

cc_library(
    name = "canonicalize_value",
    srcs = ["canonicalize_value.cc"],
    hdrs = ["canonicalize_value.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "canonicalize_value_test",
    srcs = ["canonicalize_value_test.cc"],
    deps = [
        ":canonicalize_value",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "devicedb",
    srcs = [
        "devicedb.cc",
    ],
    hdrs = [
        "devicedb.h",
        "variables.h",
    ],
    deps = [
        ":database_fbs",
    ],
)

cc_binary(
    name = "json_to_fb",
    srcs = ["json_to_fb.cc"],
    deps = [
        "//tensorflow/lite/tools:command_line_flags",
        "@flatbuffers",
    ],
)

genrule(
    name = "devicedb-sample_bin",
    srcs = [
        "database.fbs",
        "devicedb-sample.json",
    ],
    outs = ["devicedb-sample.bin"],
    cmd = """
    $(location :json_to_fb) \
        --fbs=$(location :database.fbs) \
        --json_input=$(location :devicedb-sample.json) \
        --fb_output=$(@)
    """,
    tools = [":json_to_fb"],
)

py_binary(
    name = "convert_binary_to_cc_source",
    srcs = ["convert_binary_to_cc_source.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

genrule(
    name = "devicedb-sample_cc",
    srcs = ["devicedb-sample.bin"],
    outs = [
        "devicedb-sample.cc",
        "devicedb-sample.h",
    ],
    # convert_file_to_c_source for some reason doesn't define the global with
    # 'extern', which is needed for global const variables in C++.
    cmd = """
    $(location :convert_binary_to_cc_source) \
        --input_binary_file $(location :devicedb-sample.bin) \
        --output_header_file $(location :devicedb-sample.h) \
        --output_source_file $(location :devicedb-sample.cc) \
        --array_variable_name g_tflite_acceleration_devicedb_sample_binary
    """,
    tools = [":convert_binary_to_cc_source"],
)

cc_library(
    name = "devicedb_sample",
    srcs = ["devicedb-sample.cc"],
    hdrs = ["devicedb-sample.h"],
    deps = [":database_fbs"],
)

cc_test(
    name = "devicedb_test",
    srcs = [
        "devicedb_test.cc",
    ],
    deps = [
        ":database_fbs",
        ":devicedb",
        ":devicedb_sample",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

exports_files(["gpu_compatibility.bin"])

genrule(
    name = "gpu_compatibility_binary",
    srcs = ["gpu_compatibility.bin"],
    outs = [
        "gpu_compatibility_binary.h",
        "gpu_compatibility_binary.cc",
    ],
    # convert_file_to_c_source for some reason doesn't define the global with
    # 'extern', which is needed for global const variables in C++.
    cmd = """
    $(location :convert_binary_to_cc_source) \
        --input_binary_file $(location :gpu_compatibility.bin) \
        --output_header_file $(location :gpu_compatibility_binary.h) \
        --output_source_file $(location :gpu_compatibility_binary.cc) \
        --array_variable_name g_tflite_acceleration_gpu_compatibility_binary
    """,
    tools = [":convert_binary_to_cc_source"],
)

cc_library(
    name = "android_info",
    srcs = ["android_info.cc"],
    hdrs = ["android_info.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "gpu_compatibility",
    srcs = [
        "gpu_compatibility.cc",
        "gpu_compatibility_binary.cc",
        "gpu_compatibility_binary.h",
    ],
    hdrs = [
        "gpu_compatibility.h",
    ],
    deps = [
        ":canonicalize_value",
        ":android_info",
        ":database_fbs",
        ":devicedb",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
    ] + tflite_extra_gles_deps(),
)

cc_test(
    name = "gpu_compatibility_test",
    srcs = ["gpu_compatibility_test.cc"],
    tags = tf_gpu_tests_tags() + [
        "no_cuda_asan",  # TODO(b/181032551).
    ],
    deps = [
        ":devicedb_sample",
        ":gpu_compatibility",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
