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
    name = "llvm_gpu_backend",
    hdrs = ["gpu_backend_lib.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering"],
)

cc_library(
    name = "ptx_version_util",
    hdrs = ["ptx_version_util.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:ptx_version_util"],
)

cc_library(
    name = "nvptx_backend",
    hdrs = ["nvptx_backend.h"],
    tags = [
        "cuda-only",
        "gpu",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:nvptx_backend"],
)

cc_library(
    name = "amdgpu_backend",
    hdrs = ["amdgpu_backend.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:amdgpu_backend"],
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

cc_library(
    name = "nvptx_utils",
    hdrs = ["nvptx_utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:nvptx_utils"],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/gpu/target_lowering:utils"],
)

cc_library(
    name = "spirv_backend",
    hdrs = ["spirv_backend.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ] + if_google([
        "avoid_dep",
        # TODO(b/456585142): Currently we don't support building the SYCL backend.
        "notap",
        "nobuilder",
        "manual",
    ]),
    deps = ["//xla/backends/gpu/target_lowering:spirv_backend"],
)
