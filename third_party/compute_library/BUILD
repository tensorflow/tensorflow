exports_files(["LICENSE"])

cc_library(
    name = "include",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    strip_include_prefix = "include",
    includes = ["include"],
)

cc_library(
    name = "arm_compute_core",

    srcs = glob(["src/core/*.cpp",
                 "src/core/helpers/*.cpp",
                 "src/core/CPP/**/*.cpp",
                 "src/core/utils/**/*.cpp",
                 "src/core/NEON/kernels/**/*.cpp",
                 "src/core/cpu/kernels/*.cpp",
                 "src/core/cpu/kernels/**/*.cpp",
                 "src/core/**/*.hpp",
                 "**/*.h"],
                 exclude=["src/core/TracePoint.cpp"]),

    hdrs = glob(["arm_compute/core/**/*.h",
                 "**/*.inl"])
           + ["arm_compute_version.embed"],

    includes = ["src/core/NEON/kernels/convolution/common",
                "src/core/NEON/kernels/convolution/winograd",
                "src/core/NEON/kernels/assembly"],

    defines = ["ENABLE_FP16_KERNELS", "ENABLE_FP32_KERNELS", "ENABLE_QASYMM8_KERNELS",
               "ENABLE_QASYMM8_SIGNED_KERNELS", "ENABLE_QSYMM16_KERNELS",
               "ENABLE_INTEGER_KERNELS", "ENABLE_NHWC_KERNELS"],

    deps = ["include"],
)

cc_library(
    name = "arm_compute_runtime",

    srcs = glob(["src/runtime/*.cpp",
                 "src/runtime/CPP/**/*.cpp",
                 "src/runtime/NEON/**/*.cpp",
                 "src/runtime/cpu/**/*.cpp",
                 "**/*.h"]),

    hdrs = glob(["arm_compute/runtime/**/*.h"])
            + ["arm_compute_version.embed"],

    deps = ["arm_compute_core"],

    defines = ["ARM_COMPUTE_CPP_SCHEDULER"],

    linkopts = ["-lpthread"],

    visibility = ["//visibility:public"],
)

cc_library(
    name = "arm_compute_graph",

    srcs = glob(["src/graph/*.cpp",
                 "src/graph/algorithms/*.cpp",
                 "src/graph/backends/*.cpp",
                 "src/graph/detail/*.cpp",
                 "src/graph/frontend/*.cpp",
                 "src/graph/mutators/*.cpp",
                 "src/graph/nodes/*.cpp",
                 "src/graph/printers/*.cpp",
                 "src/graph/backends/NEON/*.cpp",
                 "**/*.h"]),

    hdrs = glob(["arm_compute/graph/**/*.h"]),

    deps = ["arm_compute_core"],

    visibility = ["//visibility:public"],
)