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

# Overlay BUILD for @roc_mori//src/application. Symlinked over the extracted
# tarball's src/application/BUILD.bazel by tf_http_archive (see workspace.bzl).
#
# Mirrors src/application/CMakeLists.txt: a single cc_library named
# mori_application. All sources are host C++ (CMake sets LANGUAGE CXX on
# them); they just call into the ROCm runtime APIs (HIP, HSA, rocm-smi,
# hsakmt) plus ibverbs/libpci. No device kernels live here.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mori_application",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "bootstrap/mpi_bootstrap.cpp",
            "bootstrap/torch_bootstrap.cpp",
        ],
    ),
    copts = [
        # @local_config_rocm's rocm_headers_includes target propagates
        # -D__HIP_DISABLE_CPP_FUNCTIONS__=1 to every consumer, which hides
        # the templated hipMalloc(T**, size_t) overload.
        "-U__HIP_DISABLE_CPP_FUNCTIONS__",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        "@roc_mori//:mori_application_headers",
        # symmetric_memory.cpp includes mori/shmem/internal.hpp.
        "@roc_mori//:mori_shmem_headers",
        # CMake hip::host: libamdhip64.so + HIP host headers.
        "@local_config_rocm//rocm:hip",
        # CMake find_library(ROCM_SMI_LIB rocm_smi64): librocm_smi64.so.
        "@local_config_rocm//rocm:rocm_smi",
        "@local_config_rocm//rocm:hsa_runtime",
        "@local_config_rocm//rocm:hsakmt",
        # CMake ibverbs: system libibverbs.so (rdma-core).
        "@roc_mori//:ibverbs",
        # CMake pci: system libpci.so (pciutils). pci.cpp uses pci_alloc / pci_init
        # / pci_scan_bus / pci_read_byte.
        "@roc_mori//:libpci",
        # System libdrm + libdrm_amdgpu. Required transitively by libhsakmt.a
        # (amdgpu_get_marketing_name, amdgpu_query_gpu_info, amdgpu_*, drmClose).
        "@roc_mori//:libdrm",
        # System libnuma. Required transitively by libhsakmt.a (numa_available,
        # numa_max_node, numa_bitmask_*, mbind, numa_node_size64).
        "@roc_mori//:libnuma",
        # mori_logging interface lib in CMake is spdlog::spdlog_header_only.
        "@spdlog",
    ],
)
