# Copyright 2026 The OpenXLA Authors.
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

# Forwarding targets for libraries moved to //xla/backends/gpu/target_lowering.
#
# TODO: Remove once external users (e.g. JAX Mosaic GPU) have migrated to
# //xla/backends/gpu/target_lowering.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

cc_library(
    name = "load_ir_module",
    hdrs = ["load_ir_module.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:load_ir_module"],
)

cc_library(
    name = "nvptx_libdevice_path",
    hdrs = ["nvptx_libdevice_path.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:nvptx_libdevice_path"],
)
