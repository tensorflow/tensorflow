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
# =============================================================================

"""Module extension for llvm."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

def _llvm_zlib_compat_impl(repository_ctx):
    repository_ctx.file("BUILD", """
alias(
    name = "zlib-ng",
    actual = "@zlib//:zlib",
    visibility = ["//visibility:public"],
)
""")

_llvm_zlib_compat = repository_rule(
    implementation = _llvm_zlib_compat_impl,
)

def _llvm_zstd_compat_impl(repository_ctx):
    repository_ctx.file("BUILD", """
alias(
    name = "zstd",
    actual = "@net_zstd//:zstd",
    visibility = ["//visibility:public"],
)
""")

_llvm_zstd_compat = repository_rule(
    implementation = _llvm_zstd_compat_impl,
)

def _llvm_extension_impl(mctx):  # @unused
    _llvm_zlib_compat(name = "llvm_zlib")
    _llvm_zstd_compat(name = "llvm_zstd")
    llvm_configure(
        name = "llvm-project",
        targets = [
            "AArch64",
            "AMDGPU",
            "ARM",
            "NVPTX",
            "PowerPC",
            "RISCV",
            "SystemZ",
            "X86",
            "SPIRV",
        ],
    )

llvm_extension = module_extension(
    implementation = _llvm_extension_impl,
)
