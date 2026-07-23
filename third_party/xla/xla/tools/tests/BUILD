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

load("//xla:lit.bzl", "lit_test_suite")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "hlo_extractor_tests",
    srcs = ["hlo_extractor_test.hlo"],
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/tools:hlo-extractor",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_test(
    name = "hlo_expand_test",
    srcs = ["hlo_expand_test.cc"],
    data = [
        "data/cholesky.hlo",
        "data/invalid_concat.hlo",
        "data/spmd.hlo",
        "//xla/tools:hlo-expand",
    ],
    tags = [
        "nomsan",  # No msan for precompiled Nvidia binaries.
    ],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:subprocess",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
    ],
)
