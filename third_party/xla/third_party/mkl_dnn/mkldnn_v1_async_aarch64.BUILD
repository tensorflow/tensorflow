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

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

exports_files(["LICENSE"])

_DNNL_COPTS_THREADPOOL = [
    "-fopenmp-simd",
    "-fexceptions",
    "-DDNNL_ENABLE_MAX_CPU_ISA",
]

_CMAKE_COMMON_LIST = {
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "#undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_GPU_VENDOR DNNL_VENDOR_${DNNL_GPU_VENDOR}": "#define DNNL_GPU_VENDOR DNNL_VENDOR_NONE",
    "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
    "#cmakedefine DNNL_ENABLE_STACK_CHECKER": "#undef DNNL_ENABLE_STACK_CHECKER",
    "#cmakedefine ONEDNN_BUILD_GRAPH": "#undef ONEDNN_BUILD_GRAPH",
    "#cmakedefine DNNL_EXPERIMENTAL_SPARSE": "#undef DNNL_EXPERIMENTAL_SPARSE",
    "#cmakedefine DNNL_EXPERIMENTAL": "#undef DNNL_EXPERIMENTAL",
    "#cmakedefine01 DNNL_EXPERIMENTAL_GROUPED_MEMORY": "#define DNNL_EXPERIMENTAL_GROUPED_MEMORY 0",
    "#cmakedefine DNNL_SAFE_RBP": "#undef DNNL_SAFE_RBP",
    "#cmakedefine01 BUILD_TRAINING": "#define BUILD_TRAINING 1",
    "#cmakedefine01 BUILD_INFERENCE": "#define BUILD_INFERENCE 0",
    "#cmakedefine01 BUILD_PRIMITIVE_ALL": "#define BUILD_PRIMITIVE_ALL 1",
    "#cmakedefine01 BUILD_BATCH_NORMALIZATION": "#define BUILD_BATCH_NORMALIZATION 0",
    "#cmakedefine01 BUILD_BINARY": "#define BUILD_BINARY 0",
    "#cmakedefine01 BUILD_CONCAT": "#define BUILD_CONCAT 0",
    "#cmakedefine01 BUILD_CONVOLUTION": "#define BUILD_CONVOLUTION 0",
    "#cmakedefine01 BUILD_DECONVOLUTION": "#define BUILD_DECONVOLUTION 0",
    "#cmakedefine01 BUILD_ELTWISE": "#define BUILD_ELTWISE 0",
    "#cmakedefine01 BUILD_GATED_MLP": "#define BUILD_GATED_MLP 0",
    "#cmakedefine01 BUILD_GEMM_KERNELS_ALL": "#define BUILD_GEMM_KERNELS_ALL 1",
    "#cmakedefine01 BUILD_GEMM_KERNELS_NONE": "#define BUILD_GEMM_KERNELS_NONE 0",
    "#cmakedefine01 BUILD_GEMM_SSE41": "#define BUILD_GEMM_SSE41 0",
    "#cmakedefine01 BUILD_GEMM_AVX2": "#define BUILD_GEMM_AVX2 0",
    "#cmakedefine01 BUILD_GEMM_AVX512": "#define BUILD_GEMM_AVX512 0",
    "#cmakedefine01 BUILD_GROUP_NORMALIZATION": "#define BUILD_GROUP_NORMALIZATION 0",
    "#cmakedefine01 BUILD_INNER_PRODUCT": "#define BUILD_INNER_PRODUCT 0",
    "#cmakedefine01 BUILD_LAYER_NORMALIZATION": "#define BUILD_LAYER_NORMALIZATION 0",
    "#cmakedefine01 BUILD_LRN": "#define BUILD_LRN 0",
    "#cmakedefine01 BUILD_MATMUL": "#define BUILD_MATMUL 0",
    "#cmakedefine01 BUILD_POOLING": "#define BUILD_POOLING 0",
    "#cmakedefine01 BUILD_PRELU": "#define BUILD_PRELU 0",
    "#cmakedefine01 BUILD_REDUCTION": "#define BUILD_REDUCTION 0",
    "#cmakedefine01 BUILD_REORDER": "#define BUILD_REORDER 0",
    "#cmakedefine01 BUILD_RESAMPLING": "#define BUILD_RESAMPLING 0",
    "#cmakedefine01 BUILD_RNN": "#define BUILD_RNN 0",
    "#cmakedefine01 BUILD_SHUFFLE": "#define BUILD_SHUFFLE 0",
    "#cmakedefine01 BUILD_SOFTMAX": "#define BUILD_SOFTMAX 0",
    "#cmakedefine01 BUILD_SUM": "#define BUILD_SUM 0",
    "#cmakedefine01 BUILD_PRIMITIVE_CPU_ISA_ALL": "#define BUILD_PRIMITIVE_CPU_ISA_ALL 0",
    "#cmakedefine01 BUILD_SSE41": "#define BUILD_SSE41 0",
    "#cmakedefine01 BUILD_AVX2": "#define BUILD_AVX2 0",
    "#cmakedefine01 BUILD_AVX512": "#define BUILD_AVX512 0",
    "#cmakedefine01 BUILD_AMX": "#define BUILD_AMX 0",
    "#cmakedefine01 BUILD_GEN9": "#define BUILD_GEN9 0",
    "#cmakedefine01 BUILD_GEN11": "#define BUILD_GEN11 0",
    "#cmakedefine01 BUILD_SDPA": "#define BUILD_SDPA 0",
    "#cmakedefine01 BUILD_XE2": "#define BUILD_XE2 0",
    "#cmakedefine01 BUILD_XE3P": "#define BUILD_XE3P 0",
    "#cmakedefine01 BUILD_XE3": "#define BUILD_XE3 0",
    "#cmakedefine01 BUILD_XELP": "#define BUILD_XELP 0",
    "#cmakedefine01 BUILD_XEHPG": "#define BUILD_XEHPG 0",
    "#cmakedefine01 BUILD_XEHPC": "#define BUILD_XEHPC 0",
    "#cmakedefine01 BUILD_XEHP": "#define BUILD_XEHP 0",
    # SYCL specific settings
    "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
    "#cmakedefine DNNL_SYCL_HIP": "#undef DNNL_SYCL_HIP",
    "#cmakedefine DNNL_SYCL_GENERIC": "#undef DNNL_SYCL_GENERIC",
    "#cmakedefine DNNL_DISABLE_GPU_REF_KERNELS": "#undef DNNL_DISABLE_GPU_REF_KERNELS",
    "#cmakedefine DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER": "#undef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER",
    "#cmakedefine01 BUILD_PRIMITIVE_GPU_ISA_ALL": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 0",
}

_DNNL_RUNTIME_THREADPOOL = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
}

_DNNL_RUNTIME_THREADPOOL.update(_CMAKE_COMMON_LIST)

expand_template(
    name = "dnnl_config_h",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = select({
        "//conditions:default": _DNNL_RUNTIME_THREADPOOL,
    }),
    template = "include/oneapi/dnnl/dnnl_config.h.in",
)

expand_template(
    name = "dnnl_version_h",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "3",
        "@DNNL_VERSION_MINOR@": "14",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "N/A",
    },
    template = "include/oneapi/dnnl/dnnl_version.h.in",
)

expand_template(
    name = "dnnl_version_hash_h",
    out = "include/oneapi/dnnl/dnnl_version_hash.h",
    substitutions = {
        "@DNNL_VERSION_HASH@": "e0d8b940aba7ece3d4374f80e508bbcdb8f141db",
    },
    template = "include/oneapi/dnnl/dnnl_version_hash.h.in",
)

_INCLUDES_LIST = [
    "include",
    "src",
    "src/common",
    "src/cpu",
    "src/cpu/gemm",
    "third_party",
    "third_party/ittnotify",
    "third_party/spdlog",
    "third_party/xbyak_aarch64",
    "third_party/xbyak_aarch64/src",
    "third_party/xbyak_aarch64/xbyak_aarch64",
]

_TEXTUAL_HDRS_LIST = glob([
    "include/**/*",
    "include/*",
    "src/common/*.hpp",
    "src/common/**/*.h",
    "src/cpu/*.hpp",
    "src/cpu/**/*.hpp",
    "src/cpu/jit_utils/**/*.hpp",
    "third_party/**/*.h",
    "third_party/ittnotify/**/*.h",
    "third_party/xbyak_aarch64/**/*.h",
]) + [
    ":dnnl_config_h",
    ":dnnl_version_h",
    ":dnnl_version_hash_h",
]

alias(
    name = "mkl_dnn",
    actual = ":onednn_cpu",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "onednn_cpu",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
            "src/cpu/jit_utils/**/*.cpp",
            "third_party/ittnotify/*.c",
            "third_party/ittnotify/ittnotify/*.c",
            "third_party/xbyak_aarch64/src/*.cpp",
        ],
        exclude = [
            "src/cpu/aarch64/acl_*.cpp",
            "src/cpu/aarch64/matmul/acl_*.cpp",
            "src/cpu/aarch64/reorder/acl_*.cpp",
            "src/cpu/ppc64/**",
            "src/cpu/rv64/**",
            "src/cpu/s390x/**",
            "src/cpu/x64/**",
            "src/cpu/sycl/**",
            "src/graph/**",
            "src/xpu/**",
        ],
    ),
    copts = select({
        "//conditions:default": _DNNL_COPTS_THREADPOOL,
    }),
    includes = _INCLUDES_LIST,
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
)
