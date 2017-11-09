# Description:
#    Libraries for helping construct LLVM IR for XLA backends.

licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "alias_analysis",
    srcs = ["alias_analysis.cc"],
    hdrs = ["alias_analysis.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "llvm_util",
    srcs = ["llvm_util.cc"],
    hdrs = ["llvm_util.h"],
    deps = [
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "@llvm//:core",
        "@llvm//:support",
        "@llvm//:target",
    ],
)

cc_library(
    name = "ir_array",
    srcs = ["ir_array.cc"],
    hdrs = ["ir_array.h"],
    deps = [
        ":llvm_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "llvm_loop",
    srcs = ["llvm_loop.cc"],
    hdrs = ["llvm_loop.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "loop_emitter",
    srcs = ["loop_emitter.cc"],
    hdrs = ["loop_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_loop",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "fused_ir_emitter",
    srcs = ["fused_ir_emitter.cc"],
    hdrs = ["fused_ir_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        ":loop_emitter",
        ":tuple_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_library(
    name = "ops",
    srcs = ["ops.cc"],
    hdrs = ["ops.h"],
    deps = [
        ":fused_ir_emitter",
        ":ir_array",
        ":llvm_util",
        ":loop_emitter",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/gpu:parallel_loop_emitter",
        "//tensorflow/compiler/xla/service/gpu:partition_assignment",
    ],
)

cc_library(
    name = "tuple_ops",
    srcs = ["tuple_ops.cc"],
    hdrs = ["tuple_ops.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
