load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

exports_files(["LICENSE"])

_DNNL_COPTS_THREADPOOL = [
    "-fopenmp-simd",
    "-fexceptions",
    "-UUSE_MKL",
    "-UUSE_CBLAS",
]

_DNNL_RUNTIME_THREADPOOL = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "#undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
    "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
    "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
    "#cmakedefine DNNL_SYCL_HIP": "#undef DNNL_SYCL_HIP",
    "#cmakedefine DNNL_ENABLE_STACK_CHECKER": "#undef DNNL_ENABLE_STACK_CHECKER",
    "#cmakedefine DNNL_EXPERIMENTAL_SPARSE": "#define DNNL_EXPERIMENTAL_SPARSE",
    "#cmakedefine DNNL_EXPERIMENTAL": "#undef DNNL_EXPERIMENTAL",
    "#cmakedefine ONEDNN_BUILD_GRAPH": "#undef ONEDNN_BUILD_GRAPH",
    "#cmakedefine01 BUILD_TRAINING": "#define BUILD_TRAINING 1",
    "#cmakedefine01 BUILD_INFERENCE": "#define BUILD_INFERENCE 0",
    "#cmakedefine01 BUILD_PRIMITIVE_ALL": "#define BUILD_PRIMITIVE_ALL 1",
    "#cmakedefine01 BUILD_BATCH_NORMALIZATION": "#define BUILD_BATCH_NORMALIZATION 0",
    "#cmakedefine01 BUILD_BINARY": "#define BUILD_BINARY 0",
    "#cmakedefine01 BUILD_CONCAT": "#define BUILD_CONCAT 0",
    "#cmakedefine01 BUILD_CONVOLUTION": "#define BUILD_CONVOLUTION 0",
    "#cmakedefine01 BUILD_DECONVOLUTION": "#define BUILD_DECONVOLUTION 0",
    "#cmakedefine01 BUILD_ELTWISE": "#define BUILD_ELTWISE 0",
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
    "#cmakedefine01 BUILD_PRIMITIVE_GPU_ISA_ALL": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 0",
    "#cmakedefine01 BUILD_GEN9": "#define BUILD_GEN9 0",
    "#cmakedefine01 BUILD_GEN11": "#define BUILD_GEN11 0",
    "#cmakedefine01 BUILD_XELP": "#define BUILD_XELP 0",
    "#cmakedefine01 BUILD_XEHPG": "#define BUILD_XEHPG 0",
    "#cmakedefine01 BUILD_XEHPC": "#define BUILD_XEHPC 0",
    "#cmakedefine01 BUILD_XEHP": "#define BUILD_XEHP 0",
    "#cmakedefine01 BUILD_GROUP_NORMALIZATION": "#define BUILD_GROUP_NORMALIZATION 0",
    "#cmakedefine01 BUILD_GEMM_KERNELS_ALL": "#define BUILD_GEMM_KERNELS_ALL 1",
    "#cmakedefine01 BUILD_GEMM_KERNELS_NONE": "#define BUILD_GEMM_KERNELS_NONE 0",
    "#cmakedefine01 BUILD_GEMM_SSE41": "#define BUILD_GEMM_SSE41 0",
    "#cmakedefine01 BUILD_GEMM_AVX2": "#define BUILD_GEMM_AVX2 0",
    "#cmakedefine01 BUILD_GEMM_AVX512": "#define BUILD_GEMM_AVX512 0",
    "#cmakedefine DNNL_GPU_VENDOR": "#define DNNL_GPU_VENDOR INTEL",
    "#cmakedefine DNNL_SYCL_GENERIC": "#undef DNNL_SYCL_GENERIC",
    "#cmakedefine DNNL_DISABLE_GPU_REF_KERNELS": "#undef DNNL_DISABLE_GPU_REF_KERNELS",
    "#cmakedefine01 BUILD_SDPA": "#define BUILD_SDPA 0",
    "#cmakedefine01 BUILD_XE2": "#define BUILD_XE2 0",
    "#cmakedefine01 BUILD_XE3": "#define BUILD_XE3 0",
}

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
        "@DNNL_VERSION_MINOR@": "7",
        "@DNNL_VERSION_PATCH@": "0",
    },
    template = "include/oneapi/dnnl/dnnl_version.h.in",
)

expand_template(
    name = "dnnl_version_hash_h",
    out = "include/oneapi/dnnl/dnnl_version_hash.h",
    substitutions = {
        "@DNNL_VERSION_HASH@": "N/A",
    },
    template = "include/oneapi/dnnl/dnnl_version_hash.h.in",
)

cc_library(
    name = "mkl_dnn_acl",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/**/*.cpp",
            "src/cpu/*.cpp",
        ],
        exclude = [
            "src/cpu/x64/**",
            "src/cpu/rv64/**",
            "src/cpu/sycl/**",
            "src/xpu/**",
        ],
    ),
    copts = select({
        "//conditions:default": _DNNL_COPTS_THREADPOOL,
    }),
    defines = ["DNNL_AARCH64_USE_ACL=1"],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/aarch64/xbyak_aarch64/src",
        "src/cpu/aarch64/xbyak_aarch64/xbyak_aarch64",
        "src/cpu/gemm",
    ],
    textual_hdrs = glob(
        [
            "include/**/*",
            "include/*",
            "src/common/*.hpp",
            "src/common/**/*.h",
            "src/cpu/**/*.hpp",
            "src/cpu/*.hpp",
            "src/cpu/aarch64/xbyak_aarch64/**/*.h",
        ],
    ) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
        ":dnnl_version_hash_h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@compute_library//:arm_compute",
    ],
)
