# Utilities for building XLA computations.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//tensorflow/compiler/tf2xla:friends"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")

cc_library(
    name = "batch_dot",
    srcs = ["batch_dot.cc"],
    hdrs = ["batch_dot.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "cholesky",
    srcs = ["cholesky.cc"],
    hdrs = ["cholesky.h"],
    deps = [
        ":batch_dot",
        ":triangular_solve",
        ":util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "triangular_solve",
    srcs = ["triangular_solve.cc"],
    hdrs = ["triangular_solve.h"],
    deps = [
        ":batch_dot",
        ":util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/core:lib",
    ],
)

xla_test(
    name = "triangular_solve_test",
    srcs = ["triangular_solve_test.cc"],
    deps = [
        ":triangular_solve",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/core:lib",
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
