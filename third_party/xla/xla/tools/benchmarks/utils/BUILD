load("//xla:py_strict.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

# Python library for the matrix generation script for microbenchmarks.
py_strict_library(
    name = "generate_benchmark_matrices_lib",
    srcs = ["generate_benchmark_matrices.py"],
    deps = [
        "//third_party/py/google/protobuf",
        "//xla/tools/benchmarks/proto:benchmark_config_py_pb2",
    ],
)

# Python binary for the matrix generation script for microbenchmarks.
py_strict_binary(
    name = "generate_benchmark_matrices",
    srcs = ["generate_benchmark_matrices.py"],
    main = "generate_benchmark_matrices.py",
    deps = [":generate_benchmark_matrices_lib"],
)

# Python test for the matrix generation script for microbenchmarks.
py_strict_test(
    name = "generate_benchmark_matrices_test",
    srcs = ["generate_benchmark_matrices_test.py"],
    deps = [
        ":generate_benchmark_matrices_lib",
        "//testing/pybase",
        "//third_party/py/google/protobuf",
        "//third_party/py/more_itertools",
        "//xla/tools/benchmarks/proto:benchmark_config_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
