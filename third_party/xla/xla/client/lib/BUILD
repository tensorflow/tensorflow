# Common computation builders for XLA.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "generate_backend_suites")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

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

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:arithmetic
# instead.
cc_library(
    name = "arithmetic",
    hdrs = ["arithmetic.h"],
    deps = [
        "//xla/hlo/builder/lib:arithmetic",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:comparators
# instead.
cc_library(
    name = "comparators",
    hdrs = [
        "comparators.h",
    ],
    deps = [
        "//xla/hlo/builder/lib:comparators",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:constants
# instead.
cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = [
        "//xla/hlo/builder/lib:constants",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:broadcast
# instead.
cc_library(
    name = "broadcast",
    hdrs = ["broadcast.h"],
    deps = [
        "//xla/hlo/builder/lib:broadcast",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:conv_grad_size_util
# instead.
cc_library(
    name = "conv_grad_size_util",
    hdrs = ["conv_grad_size_util.h"],
    deps = [
        "//xla/hlo/builder/lib:conv_grad_size_util",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:dynamic_shaped_ops
# instead.
cc_library(
    name = "dynamic_shaped_ops",
    hdrs = ["dynamic_shaped_ops.h"],
    deps = [
        "//xla/hlo/builder/lib:dynamic_shaped_ops",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:loops
# instead.
cc_library(
    name = "loops",
    hdrs = ["loops.h"],
    deps = [
        "//xla/hlo/builder/lib:loops",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:math
# instead.
cc_library(
    name = "math",
    hdrs = [
        "math.h",
    ],
    deps = [
        "//xla/hlo/builder/lib:math",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:matrix
# instead.
cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    deps = [
        "//xla/hlo/builder/lib:matrix",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:pooling
# instead.
cc_library(
    name = "pooling",
    hdrs = ["pooling.h"],
    deps = [
        "//xla/hlo/builder/lib:pooling",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:prng
# instead.
cc_library(
    name = "prng",
    hdrs = ["prng.h"],
    deps = [
        "//xla/hlo/builder/lib:prng",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:qr
# instead.
cc_library(
    name = "qr",
    hdrs = ["qr.h"],
    deps = [
        "//xla/hlo/builder/lib:qr",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:lu_decomposition
# instead.
cc_library(
    name = "lu_decomposition",
    hdrs = ["lu_decomposition.h"],
    deps = [
        "//xla/hlo/builder/lib:lu_decomposition",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:approx_topk
# instead.
cc_library(
    name = "approx_topk",
    hdrs = ["approx_topk.h"],
    deps = [
        "//xla/hlo/builder/lib:approx_topk",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:approx_topk_shape
# instead.
cc_library(
    name = "approx_topk_shape",
    hdrs = ["approx_topk_shape.h"],
    deps = ["//xla/hlo/builder/lib:approx_topk_shape"],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:slicing
# instead.
cc_library(
    name = "slicing",
    hdrs = ["slicing.h"],
    deps = [
        "//xla/hlo/builder/lib:slicing",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:sorting
# instead.
cc_library(
    name = "sorting",
    hdrs = ["sorting.h"],
    deps = [
        "//xla/hlo/builder/lib:sorting",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:quantize
# instead.
cc_library(
    name = "quantize",
    hdrs = ["quantize.h"],
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

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:self_adjoint_eig
# instead.
cc_library(
    name = "self_adjoint_eig",
    hdrs = ["self_adjoint_eig.h"],
    deps = [
        "//xla/hlo/builder/lib:self_adjoint_eig",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:svd
# instead.
cc_library(
    name = "svd",
    hdrs = ["svd.h"],
    deps = [
        "//xla/hlo/builder/lib:svd",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:tridiagonal
# instead.
cc_library(
    name = "tridiagonal",
    hdrs = ["tridiagonal.h"],
    deps = [
        "//xla/hlo/builder/lib:tridiagonal",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:logdet
# instead.
cc_library(
    name = "logdet",
    deps = [
        "//xla/hlo/builder/lib:logdet",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/builder/lib:tuple
# instead.
cc_library(
    name = "tuple",
    hdrs = ["tuple.h"],
    deps = [
        "//xla/hlo/builder/lib:tuple",
    ],
)
