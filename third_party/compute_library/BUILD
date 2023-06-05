load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

exports_files(["LICENSE"])

cc_library(
    name = "include",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
    strip_include_prefix = "include",
)

_COMPUTE_LIBRARY_DEFINES = [
    "ARM_COMPUTE_OPENMP_SCHEDULER",
    "ARM_COMPUTE_CPU_ENABLED",
    "ENABLE_NEON",
    "ARM_COMPUTE_ENABLE_NEON",
    "ENABLE_SVE",
    "ARM_COMPUTE_ENABLE_SVE",
    "ARM_COMPUTE_ENABLE_BF16",
    "ARM_COMPUTE_ENABLE_I8MM",
    "ARM_COMPUTE_ENABLE_SVEF32MM",
    "ENABLE_FP32_KERNELS",
    "ENABLE_QASYMM8_KERNELS",
    "ENABLE_QASYMM8_SIGNED_KERNELS",
    "ENABLE_QSYMM16_KERNELS",
    "ENABLE_INTEGER_KERNELS",
    "ENABLE_NHWC_KERNELS",
    "ENABLE_NCHW_KERNELS",
    "ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS",
]

cc_library(
    name = "arm_compute_sve2",
    srcs = glob(
        [
            "src/cpu/kernels/**/sve2/*.cpp",
            "**/*.h",
            "**/*.hpp",
            "**/*.inl",
        ],
    ),
    copts = [
        "-march=armv8.6-a+sve2",
        "-fopenmp",
    ],
    defines = _COMPUTE_LIBRARY_DEFINES + ["ARM_COMPUTE_ENABLE_SVE2"],
    includes = [
        "src/core/NEON/kernels/arm_conv",
        "src/core/NEON/kernels/arm_gemm",
        "src/core/NEON/kernels/assembly",
        "src/core/cpu/kernels/assembly",
        "src/cpu/kernels/assembly",
    ],
    linkopts = ["-fopenmp"],
    deps = ["include"],
)

cc_library(
    name = "arm_compute_sve",
    srcs = glob(
        [
            "src/core/NEON/kernels/arm_gemm/kernels/sve_*/*.cpp",
            "src/core/NEON/kernels/arm_conv/**/kernels/sve_*/*.cpp",
            "src/core/NEON/kernels/arm_conv/depthwise/interleaves/sve_*.cpp",
            "src/core/NEON/kernels/batchnormalization/impl/SVE/*.cpp",
            "src/core/NEON/kernels/convolution/winograd/input_transforms/sve_fp32_6x6.cpp",
            "src/cpu/kernels/**/sve/*.cpp",
            "**/*.h",
            "**/*.hpp",
            "**/*.inl",
        ],
    ) + [
        "src/core/NEON/kernels/arm_gemm/mergeresults-sve.cpp",
        "src/core/NEON/kernels/arm_gemm/transform-sve.cpp",
    ],
    copts = [
        "-march=armv8.2-a+sve",
        "-fopenmp",
    ],
    defines = _COMPUTE_LIBRARY_DEFINES,
    includes = [
        "src/core/NEON/kernels/arm_conv",
        "src/core/NEON/kernels/arm_gemm",
        "src/core/NEON/kernels/assembly",
        "src/core/cpu/kernels/assembly",
        "src/cpu/kernels/assembly",
    ],
    linkopts = ["-fopenmp"],
    deps = ["include"],
)

cc_library(
    name = "arm_compute",
    srcs = glob(
        [
            "src/common/**/*.cpp",
            "src/core/*.cpp",
            "src/core/CPP/kernels/*.cpp",
            "src/core/helpers/*.cpp",
            "src/core/utils/**/*.cpp",
            "src/runtime/**/*.cpp",
            "src/c/*.cpp",
            "src/core/NEON/kernels/*.cpp",
            "src/core/NEON/kernels/convolution/**/*.cpp",
            "src/core/NEON/kernels/arm_gemm/kernels/a64_*/*.cpp",
            "src/core/NEON/kernels/arm_conv/pooling/*.cpp",
            "src/core/NEON/kernels/arm_conv/**/kernels/a64_*/*.cpp",
            "src/core/NEON/kernels/arm_conv/depthwise/*.cpp",
            "src/core/NEON/kernels/arm_conv/depthwise/interleaves/a64_*.cpp",
            "src/core/NEON/kernels/arm_conv/depthwise/interleaves/generic*.cpp",
            "src/core/NEON/kernels/batchnormalization/impl/NEON/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/kernels/*.cpp",
            "src/cpu/kernels/fuse_batch_normalization/**/*.cpp",
            "src/cpu/kernels/*/generic/*.cpp",
            "src/cpu/operators/**/*.cpp",
            "src/cpu/utils/*.cpp",
            "src/cpu/kernels/internal/*.cpp",
            "src/cpu/kernels/**/neon/*.cpp",
            "src/cpu/kernels/**/nchw/*.cpp",
            "src/core/NEON/kernels/arm_gemm/*.cpp",
            "**/*.h",
            "**/*.hpp",
            "**/*.inl",
        ],
        exclude = [
            "src/core/utils/logging/**",
            "src/core/TracePoint.cpp",
            "src/core/NEON/kernels/arm_gemm/mergeresults-sve.cpp",
            "src/core/NEON/kernels/arm_gemm/transform-sve.cpp",
            "src/core/NEON/kernels/convolution/winograd/input_transforms/sve_fp32_6x6.cpp",
            "src/runtime/CL/**",
            "src/gpu/**",
        ],
    ) + [
        "src/c/operators/AclActivation.cpp",
        "src/core/CPP/CPPTypes.cpp",
        "src/core/NEON/kernels/arm_conv/addressing.cpp",
        "src/core/NEON/kernels/arm_conv/depthwise/interleaves/8b_mla.cpp",
        "src/core/NEON/kernels/arm_conv/pooling/kernels/cpp_nhwc_1x1_stride_any_depthfirst/generic.cpp",
    ],
    hdrs = glob([
        "src/core/NEON/kernels/**/*.h",
        "src/core/NEON/kernels/**/*.hpp",
        "arm_compute/runtime/**/*.h",
        "arm_compute/runtime/*.h",
        "arm_compute/core/**/*.h",
        "**/*.inl",
    ]) + [
        "arm_compute_version.embed",
    ],
    copts = [
        "-march=armv8-a",
        "-fopenmp",
    ],
    defines = _COMPUTE_LIBRARY_DEFINES,
    includes = [
        "arm_compute/runtime",
        "src/core/NEON/kernels/assembly",
        "src/core/NEON/kernels/convolution/common",
        "src/core/NEON/kernels/convolution/winograd",
        "src/core/cpu/kernels/assembly",
        "src/cpu/kernels/assembly",
    ],
    linkopts = ["-fopenmp"],
    visibility = ["//visibility:public"],
    deps = [
        "arm_compute_sve",
        "arm_compute_sve2",
        "include",
    ],
)

config_setting(
    name = "build_with_acl",
    define_values = {
        "build_with_acl": "true",
    },
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
