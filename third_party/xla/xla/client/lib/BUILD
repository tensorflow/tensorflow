# Common computation builders for XLA.

load("//xla/tests:build_defs.bzl", "generate_backend_suites")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/client:friends"]),
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

cc_library(
    name = "arithmetic",
    hdrs = ["arithmetic.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:arithmetic instead.",
    deps = [
        "//xla/hlo/builder/lib:arithmetic",
    ],
)

cc_library(
    name = "comparators",
    hdrs = [
        "comparators.h",
    ],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:comparators instead.",
    deps = [
        "//xla/hlo/builder/lib:comparators",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:constants instead.",
    deps = [
        "//xla/hlo/builder/lib:constants",
    ],
)

cc_library(
    name = "broadcast",
    hdrs = ["broadcast.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:broadcast instead.",
    deps = [
        "//xla/hlo/builder/lib:broadcast",
    ],
)

cc_library(
    name = "conv_grad_size_util",
    hdrs = ["conv_grad_size_util.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:conv_grad_size_util instead.",
    deps = [
        "//xla/hlo/builder/lib:conv_grad_size_util",
    ],
)

cc_library(
    name = "dynamic_shaped_ops",
    hdrs = ["dynamic_shaped_ops.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:dynamic_shaped_ops instead.",
    deps = [
        "//xla/hlo/builder/lib:dynamic_shaped_ops",
    ],
)

cc_library(
    name = "loops",
    hdrs = ["loops.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:loops instead.",
    deps = [
        "//xla/hlo/builder/lib:loops",
    ],
)

cc_library(
    name = "math",
    hdrs = [
        "math.h",
    ],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:math instead.",
    deps = [
        "//xla/hlo/builder/lib:math",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:matrix instead.",
    deps = [
        "//xla/hlo/builder/lib:matrix",
    ],
)

cc_library(
    name = "pooling",
    hdrs = ["pooling.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:pooling instead.",
    deps = [
        "//xla/hlo/builder/lib:pooling",
    ],
)

cc_library(
    name = "prng",
    hdrs = ["prng.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:prng instead.",
    deps = [
        "//xla/hlo/builder/lib:prng",
    ],
)

cc_library(
    name = "qr",
    hdrs = ["qr.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:qr instead.",
    deps = [
        "//xla/hlo/builder/lib:qr",
    ],
)

cc_library(
    name = "lu_decomposition",
    hdrs = ["lu_decomposition.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:lu_decomposition instead.",
    deps = [
        "//xla/hlo/builder/lib:lu_decomposition",
    ],
)

cc_library(
    name = "approx_topk",
    hdrs = ["approx_topk.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:approx_topk instead.",
    deps = [
        "//xla/hlo/builder/lib:approx_topk",
    ],
)

cc_library(
    name = "approx_topk_shape",
    hdrs = ["approx_topk_shape.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:approx_topk_shape instead.",
    deps = ["//xla/hlo/builder/lib:approx_topk_shape"],
)

cc_library(
    name = "slicing",
    hdrs = ["slicing.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:slicing instead.",
    deps = [
        "//xla/hlo/builder/lib:slicing",
    ],
)

cc_library(
    name = "sorting",
    hdrs = ["sorting.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:sorting instead.",
    deps = [
        "//xla/hlo/builder/lib:sorting",
    ],
)

cc_library(
    name = "quantize",
    hdrs = ["quantize.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:quantize instead.",
    deps = [
        "//xla/hlo/builder/lib:quantize",
    ],
)

cc_library(
    name = "testing",
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client",
        "//xla/client:global_data",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "self_adjoint_eig",
    hdrs = ["self_adjoint_eig.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:self_adjoint_eig instead.",
    deps = [
        "//xla/hlo/builder/lib:self_adjoint_eig",
    ],
)

cc_library(
    name = "svd",
    hdrs = ["svd.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:svd instead.",
    deps = [
        "//xla/hlo/builder/lib:svd",
    ],
)

cc_library(
    name = "tridiagonal",
    hdrs = ["tridiagonal.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:tridiagonal instead.",
    deps = [
        "//xla/hlo/builder/lib:tridiagonal",
    ],
)

cc_library(
    name = "logdet",
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:logdet instead.",
    deps = [
        "//xla/hlo/builder/lib:logdet",
    ],
)

cc_library(
    name = "tuple",
    hdrs = ["tuple.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/builder/lib:tuple instead.",
    deps = [
        "//xla/hlo/builder/lib:tuple",
    ],
)
