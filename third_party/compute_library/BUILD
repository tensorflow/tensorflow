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

cc_library(
    name = "arm_compute_core",
    srcs = glob(
        [
            "src/core/*.cpp",
            "src/core/helpers/*.cpp",
            "src/core/CPP/**/*.cpp",
            "src/core/utils/**/*.cpp",
            "src/core/NEON/kernels/**/*.cpp",
            "src/core/cpu/kernels/*.cpp",
            "src/core/cpu/kernels/**/*.cpp",
            "src/core/**/*.hpp",
            "**/*.h",
        ],
        exclude = ["src/core/TracePoint.cpp"],
    ),
    hdrs = glob([
        "arm_compute/core/**/*.h",
        "**/*.inl",
    ]) + [
        "arm_compute_version.embed",
    ],
    defines = [
        "ENABLE_FP16_KERNELS",
        "ENABLE_FP32_KERNELS",
        "ENABLE_QASYMM8_KERNELS",
        "ENABLE_QASYMM8_SIGNED_KERNELS",
        "ENABLE_QSYMM16_KERNELS",
        "ENABLE_INTEGER_KERNELS",
        "ENABLE_NHWC_KERNELS",
    ],
    includes = [
        "arm_compute/runtime",
        "src/core/NEON/kernels/assembly",
        "src/core/NEON/kernels/convolution/common",
        "src/core/NEON/kernels/convolution/winograd",
    ],
    deps = ["include"],
)

cc_library(
    name = "arm_compute_runtime",
    srcs = glob([
        "src/runtime/*.cpp",
        "src/runtime/CPP/**/*.cpp",
        "src/runtime/NEON/**/*.cpp",
        "src/runtime/cpu/**/*.cpp",
        "**/*.h",
    ]),
    hdrs = glob([
        "arm_compute/runtime/**/*.h",
        "arm_compute/runtime/*.h",
    ]) + [
        "arm_compute_version.embed",
    ],
    defines = ["ARM_COMPUTE_CPP_SCHEDULER"],
    linkopts = ["-lpthread"],
    visibility = ["//visibility:public"],
    deps = ["arm_compute_core"],
)

cc_library(
    name = "arm_compute_graph",
    srcs = glob([
        "src/graph/*.cpp",
        "src/graph/algorithms/*.cpp",
        "src/graph/backends/*.cpp",
        "src/graph/detail/*.cpp",
        "src/graph/frontend/*.cpp",
        "src/graph/mutators/*.cpp",
        "src/graph/nodes/*.cpp",
        "src/graph/printers/*.cpp",
        "src/graph/backends/NEON/*.cpp",
        "**/*.h",
    ]),
    hdrs = glob(["arm_compute/graph/**/*.h"]),
    visibility = ["//visibility:public"],
    deps = ["arm_compute_core"],
)
