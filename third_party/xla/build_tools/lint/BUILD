# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

load(
    "//xla:pytype.default.bzl",
    "pytype_strict_binary",
    "pytype_strict_library",
)
# Placeholder: load py_test

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_binary(
    name = "clang_tidy",
    srcs = ["clang_tidy.py"],
    deps = [":diff_parser"],
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

py_test(
    name = "check_contents_test",
    srcs = ["check_contents_test.py"],
    data = [
        "testdata/bad_cc.diff",
        "testdata/important_cc.diff",
    ],
    tags = ["no_oss"],
    deps = [
        ":check_contents",
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
    tags = ["no_oss"],
    deps = [
        ":diff_parser",
        "@absl_py//absl/testing:absltest",
    ],
)
