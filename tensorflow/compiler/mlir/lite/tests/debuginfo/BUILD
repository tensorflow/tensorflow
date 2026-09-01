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

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":debug_info_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "pbtxt",
        #        "py", TODO(b/150304798)
    ],
)

# Bundle together all the debug info files that are used by the tests.
filegroup(
    name = "debug_info_files",
    srcs = glob(
        ["**/*.debug"],
    ),
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":concrete_function_error",
        ":saved_model_error",
        "//tensorflow/compiler/mlir/lite:flatbuffer_to_string",
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

py_binary(
    name = "saved_model_error",
    srcs = ["saved_model_error.py"],
    exec_properties = if_oss(
        None,
        select({
            "@bazel_tools//tools/cpp:asan_build": {"cpp_link.mem": "20g"},
            "//conditions:default": None,
        }),
    ),
    main = "saved_model_error.py",
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)

py_binary(
    name = "concrete_function_error",
    srcs = ["concrete_function_error.py"],
    exec_properties = if_oss(
        None,
        select({
            "@bazel_tools//tools/cpp:asan_build": {"cpp_link.mem": "20g"},
            "//conditions:default": None,
        }),
    ),
    main = "concrete_function_error.py",
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)
