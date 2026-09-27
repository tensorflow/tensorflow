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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
load("@llvm-raw//utils/bazel:linux_uapi.bzl", "linux_uapi_setup")
load("//third_party:repo.bzl", "tf_mirror_urls")

_PYYAML_CONTENT = """\
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for PyYAML)
    licenses = ["notice"],
)

py_library(
    name = "yaml",
    srcs = glob(["yaml/*.py"]),
)
"""

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
    linux_uapi_setup(name = "linux_uapi")
    http_archive(
        name = "pyyaml",
        urls = tf_mirror_urls(
            "https://github.com/yaml/pyyaml/archive/refs/tags/5.1.zip",
        ),
        sha256 = "f0a35d7f282a6d6b1a4f3f3965ef5c124e30ed27a0088efb97c0977268fd671f",
        strip_prefix = "pyyaml-5.1/lib3",
        build_file_content = _PYYAML_CONTENT,
    )
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
