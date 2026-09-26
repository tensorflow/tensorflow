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
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("//tensorflow/lite:special_rules.bzl", "gpu_compatibility_without_gl_deps_internal_visibility_allowlist", "tflite_extra_gles_deps", "tflite_portable_test_suite")
load("//tensorflow/lite/delegates/gpu:build_defs.bzl", "gpu_delegate_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
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
        ":devicedb",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "canonicalize_value_test",
    srcs = ["canonicalize_value_test.cc"],
    deps = [
        ":canonicalize_value",
        ":devicedb",
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
    $(execpath :json_to_fb) \
        --fbs=$(execpath :database.fbs) \
        --json_input=$(execpath :devicedb-sample.json) \
        --fb_output=$(@)
    """,
    tools = [":json_to_fb"],
)

py_binary(
    name = "convert_binary_to_cc_source",
    srcs = ["convert_binary_to_cc_source.py"],
    strict_deps = True,
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
    $(execpath :convert_binary_to_cc_source) \
        --input_binary_file $(execpath :devicedb-sample.bin) \
        --output_header_file $(execpath :devicedb-sample.h) \
        --output_source_file $(execpath :devicedb-sample.cc) \
        --array_variable_name g_tflite_acceleration_devicedb_sample_binary
    """,
    tools = [":convert_binary_to_cc_source"],
)

cc_library(
    name = "devicedb_sample",
    srcs = ["devicedb-sample.cc"],
    hdrs = [
        "devicedb-sample.h",
        "variables.h",
    ],
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

exports_files([
    "database.fbs",
    "gpu_compatibility.bin",
    "mldrift_compatibility.bin",
])

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
    $(execpath :convert_binary_to_cc_source) \
        --input_binary_file $(execpath :gpu_compatibility.bin) \
        --output_header_file $(execpath :gpu_compatibility_binary.h) \
        --output_source_file $(execpath :gpu_compatibility_binary.cc) \
        --array_variable_name g_tflite_acceleration_gpu_compatibility_binary
    """,
    tools = [":convert_binary_to_cc_source"],
)

genrule(
    name = "mldrift_compatibility_binary",
    srcs = ["mldrift_compatibility.bin"],
    outs = [
        "mldrift_compatibility_binary.h",
        "mldrift_compatibility_binary.cc",
    ],
    # The array variable name must differ from the gpu_compatibility one above:
    # both allowlists may end up linked into the same binary.
    cmd = """
    $(execpath :convert_binary_to_cc_source) \
        --input_binary_file $(execpath :mldrift_compatibility.bin) \
        --output_header_file $(execpath :mldrift_compatibility_binary.h) \
        --output_source_file $(execpath :mldrift_compatibility_binary.cc) \
        --array_variable_name g_tflite_acceleration_mldrift_compatibility_binary
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
    name = "gpu_compatibility_binary_embed",
    srcs = [":gpu_compatibility_binary.cc"],
    hdrs = [":gpu_compatibility_binary.h"],
)

# Deliberately a target of its own rather than an addition to the library above:
# a client that only cares about ML Drift should not link the (much larger)
# TFLite GPU allowlist, and vice versa. Both binaries are consumed through the
# generic GPUCompatibilityList::Create(flatbuffer, length) overload.
cc_library(
    name = "mldrift_compatibility_binary_embed",
    srcs = [":mldrift_compatibility_binary.cc"],
    hdrs = [":mldrift_compatibility_binary.h"],
)

cc_library(
    name = "gpu_compatibility",
    hdrs = ["gpu_compatibility.h"],
    deps = [
        ":android_info",
        ":database_fbs",
        ":devicedb_sample",
        ":gpu_compatibility_without_gl_deps",
        "//tensorflow/lite/delegates/gpu:delegate_options",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "@com_google_absl//absl/strings:string_view",
    ] + tflite_extra_gles_deps(),
)

cc_library(
    name = "gpu_compatibility_without_gl_deps",
    srcs = ["gpu_compatibility.cc"],
    hdrs = ["gpu_compatibility.h"],
    visibility = gpu_compatibility_without_gl_deps_internal_visibility_allowlist(),
    deps = [
        ":android_info",
        ":canonicalize_value",
        ":database_fbs",
        ":devicedb",
        ":gpu_compatibility_binary_embed",
        "//tensorflow/lite/delegates/gpu:delegate_options",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_test(
    name = "gpu_compatibility_test",
    srcs = ["gpu_compatibility_test.cc"],
    tags = [
        "gpu",
        "no_cuda_asan",  # TODO(b/181032551).
        "notsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":devicedb_sample",
        ":gpu_compatibility",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mldrift_compatibility",
    srcs = ["mldrift_compatibility.cc"],
    hdrs = ["mldrift_compatibility.h"],
    linkopts = gpu_delegate_linkopts(),
    deps = [
        ":android_info",
        ":gpu_compatibility",
        ":mldrift_compatibility_binary_embed",
        "//tensorflow/lite:logger",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "mldrift_compatibility_test",
    srcs = ["mldrift_compatibility_test.cc"],
    linkopts = gpu_delegate_linkopts(),
    deps = [
        ":android_info",
        ":devicedb_sample",
        ":mldrift_compatibility",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
