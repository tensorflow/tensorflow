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

# Overlay BUILD for @roc_mori//src/cco. Symlinked over the extracted tarball's
# src/cco/BUILD.bazel by tf_http_archive (see workspace.bzl).
#
# Mirrors the host-side mori_cco target in src/cco/CMakeLists.txt: a single
# host TU (cco_init.cpp) layered on top of mori_application (which CMake
# whole-archives into libmori_cco.so). The device wrapper
# device/cco_device_wrapper.cpp is excluded here — like the shmem device
# wrapper, it is JIT/prebuilt into bitcode elsewhere, not part of this lib.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mori_cco",
    srcs = ["cco_init.cpp"],
    # PUBLIC BUILD_CCO_SDMA=1 mirrors src/cco/CMakeLists.txt: it must match the
    # value every dependent that includes mori/cco/cco.hpp compiles with.
    defines = ["BUILD_CCO_SDMA=1"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        "@roc_mori//src/application:mori_application",
        "@roc_mori//:mori_application_headers",
        # symmetric_memory.cpp (pulled in transitively) includes
        # mori/shmem/internal.hpp.
        "@roc_mori//:mori_shmem_headers",
        # CMake hip::host: libamdhip64.so + HIP host headers.
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hsa_runtime",
        "@local_config_rocm//rocm:hsakmt",
        # infiniband/verbs.h via transport/rdma providers.
        "@roc_mori//:ibverbs",
        # libdrm + libdrm_amdgpu, required transitively by libhsakmt.a.
        "@roc_mori//:libdrm",
        # libnuma, required transitively by libhsakmt.a.
        "@roc_mori//:libnuma",
        # mori_logging interface lib in CMake is spdlog::spdlog_header_only.
        "@spdlog",
    ],
)
