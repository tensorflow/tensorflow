# Common computation builders for XLA.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow/compiler/xla/client:friends"])

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/client/xla_client:xla_computation",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "testing",
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/client/xla_client:xla_computation",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
    ],
)
