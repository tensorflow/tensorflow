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

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "generate_backend_suites")

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

cc_library(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        ":constants",
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
    name = "constants",
    srcs = ["constants.cc"],
    hdrs = ["constants.h"],
    deps = [
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
    ],
)

xla_test(
    name = "constants_test",
    srcs = ["constants_test.cc"],
    tags = ["enable_for_xla_interpreter"],
    deps = [
        ":constants",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "math",
    srcs = ["math.cc"],
    hdrs = ["math.h"],
    deps = [
        ":constants",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
    ],
)

xla_test(
    name = "math_test",
    srcs = ["math_test.cc"],
    tags = ["enable_for_xla_interpreter"],
    deps = [
        ":math",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "numeric",
    srcs = ["numeric.cc"],
    hdrs = ["numeric.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
    ],
)

xla_test(
    name = "numeric_test",
    srcs = ["numeric_test.cc"],
    tags = ["enable_for_xla_interpreter"],
    deps = [
        ":numeric",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "prng",
    srcs = ["prng.cc"],
    hdrs = ["prng.h"],
    deps = [
        ":constants",
        ":math",
        ":numeric",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "testing",
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:literal",
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
