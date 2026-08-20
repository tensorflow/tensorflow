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

# Overlay BUILD for @roc_mori//src/collective. Symlinked over the extracted
# tarball's src/collective/BUILD.bazel by tf_http_archive (see workspace.bzl).
#
# Mirrors the host-only mori_collective target in src/collective/CMakeLists.txt.
# The four .cpp files are pure host C++: the SDMA collective kernels are
# JIT-compiled from ccl_kernels.hip at runtime, so this target carries no
# device code (CMake links only hip::host, no -x hip / --hip-link).
#
# Sources are listed explicitly (not globbed) because core/ also holds files
# that belong to the internode lib or are unused, and must NOT be picked up.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mori_collective",
    srcs = [
        "core/allgather_into_tensor.cpp",
        "core/oneshot_all2all_sdma_class.cpp",
        "core/oneshot_allgather_sdma_class.cpp",
        "core/twoshot_allreduce_sdma_class.cpp",
    ],
    copts = [
        # @local_config_rocm's rocm_headers_includes target propagates
        # -D__HIP_DISABLE_CPP_FUNCTIONS__=1 to every consumer, which hides the
        # templated hipMalloc(T**, size_t) overload these TUs rely on.
        "-U__HIP_DISABLE_CPP_FUNCTIONS__",
        # These host TUs #include "mori/shmem/shmem.hpp", whose __device__
        # globalGpuStates static-init block must be suppressed in host-only
        # compilation.
        "-DMORI_SHMEM_NO_STATIC_INIT",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        # CMake target_link_libraries: mori_shmem, mori_application,
        # mori_logging (spdlog). The .cpp files call shmem::* and application::*
        # symbols, so the compiled libs (not just headers) are required.
        "@roc_mori//src/shmem:mori_shmem",
        "@roc_mori//src/application:mori_application",
        "@roc_mori//:mori_shmem_headers",
        "@roc_mori//:mori_application_headers",
        # CMake hip::host: libamdhip64.so + HIP host headers.
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
        # infiniband/verbs.h transitively via shmem headers.
        "@roc_mori//:ibverbs",
        "@spdlog",
    ],
)
