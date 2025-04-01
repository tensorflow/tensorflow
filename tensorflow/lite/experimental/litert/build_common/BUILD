# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

exports_files(srcs = [
    "export_litert_only_darwin.lds",
    "export_litert_only_linux.lds",
    "export_litert_runtime_only_darwin.lds",
    "export_litert_runtime_only_linux.lds",
])

bzl_library(
    name = "special_rule_bzl",
    srcs = ["special_rule.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "litert_build_defs_bzl",
    srcs = ["litert_build_defs.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "tfl_model_gen_bzl",
    srcs = ["tfl_model_gen.bzl"],
    visibility = ["//visibility:private"],
)
