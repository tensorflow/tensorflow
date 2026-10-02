# Copyright 2025 The OpenXLA Authors.
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

load("@rules_ml_toolchain//cc/sanitizers:sanitizer_wrapper.bzl", "sanitizer_wrapper")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

# Export ignore lists for use by sanitizer wrappers
exports_files([
    "asan_ignore_list.txt",
    "lsan_ignore_list.txt",
    "tsan_ignore_list.txt",
])

# Sanitizer wrapper using rules_ml_toolchain macro
# This replaces the custom genrule+sh_binary implementation
# Uses XLA's custom suppression lists for ROCm/HIP libraries
# buildifier: disable=platform-specific-binaries
sanitizer_wrapper(
    name = "sanitizer_wrapper",
    additional_data = ["@llvm_linux_x86_64//:distro_libs"],
    asan_ignore_list = ":asan_ignore_list.txt",
    # XLA-specific sanitizer options
    asan_options = "detect_leaks=0:use_sigaltstack=0",
    llvm_symbolizer = "@llvm_linux_x86_64//:llvm-symbolizer",
    lsan_ignore_list = ":lsan_ignore_list.txt",
    tsan_ignore_list = ":tsan_ignore_list.txt",
    tsan_options = "halt_on_error=1:exitcode=66:history_size=7:ignore_noninstrumented_modules=1",
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "parallel_gpu_execute",
    srcs = ["parallel_gpu_execute.sh"],
    data = [
        "@local_config_rocm//rocm:rocminfo",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Sanitizer wrapper that runs under parallel_gpu_execute
# Combines sanitizer setup with GPU locking and device visibility
# buildifier: disable=platform-specific-binaries
sanitizer_wrapper(
    name = "sanitizer_wrapper_parallel_gpu_execute",
    additional_data = [
        "@llvm_linux_x86_64//:distro_libs",
        "@local_config_rocm//rocm:rocminfo",
    ],
    asan_ignore_list = ":asan_ignore_list.txt",
    asan_options = "detect_leaks=0:use_sigaltstack=0",
    llvm_symbolizer = "@llvm_linux_x86_64//:llvm-symbolizer",
    lsan_ignore_list = ":lsan_ignore_list.txt",
    run_under = ":parallel_gpu_execute",
    tsan_ignore_list = ":tsan_ignore_list.txt",
    tsan_options = "halt_on_error=1:exitcode=66:history_size=7:ignore_noninstrumented_modules=1",
    visibility = ["//visibility:public"],
)
