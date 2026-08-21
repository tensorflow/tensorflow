# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

# Tools and utilities for analyzing the BufferDebugLogProto dumps.

load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "checksum_mismatch_report",
    srcs = ["checksum_mismatch_report.py"],
    deps = [
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_py",
        "//xla/backends/gpu/runtime:thunk_proto_py",
    ],
)

py_test(
    name = "checksum_mismatch_report_test",
    srcs = ["checksum_mismatch_report_test.py"],
    strict_deps = True,
    deps = [
        ":checksum_mismatch_report",
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_py",
        "//xla/backends/gpu/runtime:thunk_proto_py",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_binary(
    name = "check_thunk_output_consistency",
    srcs = [
        "check_thunk_output_consistency.py",
    ],
    main = "check_thunk_output_consistency.py",
    deps = [
        ":checksum_mismatch_report",
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_py",
        "//xla/backends/gpu/runtime:thunk_proto_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)
