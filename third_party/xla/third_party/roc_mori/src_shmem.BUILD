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

# Overlay BUILD for @roc_mori//src/shmem. Symlinked over the extracted
# tarball's src/shmem/BUILD.bazel by tf_http_archive (see workspace.bzl).
#
# Mirrors src/shmem/CMakeLists.txt: a single SHARED-style cc_library named
# mori_shmem. Sources are globbed so new pure-CXX files added upstream get
# picked up automatically; the one explicit exclusion is the device-code
# wrapper (CMake excludes it from MORI_SHMEM_SOURCES too — it gets JITed
# elsewhere or built into a separate target).
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mori_shmem",
    srcs = glob(
        ["*.cpp"],
        exclude = [
            # Contains __device__ symbols. Keep in sync with MORI_SHMEM_SOURCES
            # in src/shmem/CMakeLists.txt — that list excludes this file too.
            "shmem_device_api_wrapper.cpp",
        ],
    ),
    copts = [
        # @local_config_rocm's rocm_headers_includes target propagates
        # -D__HIP_DISABLE_CPP_FUNCTIONS__=1 to every consumer, which suppresses
        # the templated hipMalloc(T**, size_t) wrapper.
        "-U__HIP_DISABLE_CPP_FUNCTIONS__",
    ],
    deps = [
        "@local_config_rocm//rocm:hip",  # libamdhip64.so (hip::host)
        "@local_config_rocm//rocm:rocm_headers",
        "@roc_mori//:ibverbs",
        "@roc_mori//:mori_application_headers",
        "@roc_mori//:mori_shmem_headers",
        "@roc_mori//src/application:mori_application",
        "@spdlog",
    ],
    # Required for static linking. Some MORI shmem archive members participate
    # through runtime side effects, not only direct symbol references: HIP TUs
    # that include shmem.hpp register device-side GpuStates providers, and
    # ShmemInit later uses runtime.cpp's registry in CopyGpuStatesToDevice() to
    # populate the `globalGpuStates` symbol used by device kernels. If Bazel
    # links this archive normally, the program can still link while omitting the
    # relevant shmem runtime objects; the registry then stays empty/stale and the
    # first kernel that dereferences globalGpuStates fails with GPU illegal
    # memory access. This mirrors MORI's shared-library CMake build, where all
    # shmem objects are present in the final DSO.
    alwayslink = True,
)
