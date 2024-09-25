# Copyright 2023 The OpenXLA Authors.
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
# ============================================================================

load("@rules_python//python:py_test.bzl", "py_test")
load("//xla:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "check_contents",
    srcs = ["check_contents.py"],
    deps = [":diff_parser"],
)

pytype_strict_library(
    name = "diff_parser",
    srcs = ["diff_parser.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "generate_compile_commands",
    srcs = ["generate_compile_commands.py"],
)

py_test(
    name = "check_contents_test",
    srcs = ["check_contents_test.py"],
    data = [
        "testdata/bad_cc.diff",
        "testdata/important_cc.diff",
    ],
    deps = [
        ":check_contents",
        "//build_tools:test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "diff_parser_test",
    srcs = ["diff_parser_test.py"],
    data = [
        "testdata/bad_cc.diff",
        "testdata/crosstool.diff",
        "testdata/important_cc.diff",
    ],
    deps = [
        ":diff_parser",
        "//build_tools:test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "generate_compile_commands_test",
    srcs = ["generate_compile_commands_test.py"],
    deps = [
        ":generate_compile_commands",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_binary(
    name = "tags",
    srcs = ["tags.py"],
)
