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
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/backends/gpu/tests:helper.bzl", "get_canonical_repo_name")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform/default:build_config_root.bzl", "tf_gpu_tests_tags")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "ptx_opt",
    srcs = ["ptx_opt.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/service/gpu/llvm_gpu_backend:load_ir_module",
        "//xla/service/gpu/llvm_gpu_backend:nvptx_backend",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_platform",
    ]),
)

lit_test_suite(
    name = "ptx_opt_tests",
    # It fails in OSS to load the module correctly.
    srcs = if_google(
        glob(["**/*.ll"]),
        [],
    ),
    args = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=GPU=a100_pcie_80",
    ]),
    cfg = "//xla:lit.cfg.py",
    default_tags = tf_gpu_tests_tags(),
    hermetic_cuda_data_dir = "%S/../../../../../" + get_canonical_repo_name("cuda_nvcc"),
    tags = [
        "cuda-only",
        "gpu",
    ],
    tools = [
        ":ptx_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
