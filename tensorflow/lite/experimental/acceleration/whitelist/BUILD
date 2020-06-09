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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

flatbuffer_cc_library(
    name = "database_fbs",
    srcs = ["database.fbs"],
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
    srcs_version = "PY2AND3",
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

cc_test(
    name = "devicedb_test",
    srcs = [
        "devicedb-sample.cc",
        "devicedb-sample.h",
        "devicedb_test.cc",
    ],
    deps = [
        ":database_fbs",
        ":devicedb",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

genrule(
    name = "gpu_whitelist_binary",
    srcs = ["gpu_whitelist.bin"],
    outs = [
        "gpu_whitelist_binary.h",
        "gpu_whitelist_binary.cc",
    ],
    # convert_file_to_c_source for some reason doesn't define the global with
    # 'extern', which is needed for global const variables in C++.
    cmd = """
    $(location :convert_binary_to_cc_source) \
        --input_binary_file $(location :gpu_whitelist.bin) \
        --output_header_file $(location :gpu_whitelist_binary.h) \
        --output_source_file $(location :gpu_whitelist_binary.cc) \
        --array_variable_name g_tflite_acceleration_gpu_whitelist_binary
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
    name = "gpu_whitelist",
    srcs = [
        "gpu_whitelist.cc",
        "gpu_whitelist_binary.cc",
        "gpu_whitelist_binary.h",
    ],
    hdrs = [
        "gpu_whitelist.h",
    ],
    deps = [
        ":android_info",
        ":database_fbs",
        ":devicedb",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()
