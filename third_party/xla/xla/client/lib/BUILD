# Common computation builders for XLA.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "generate_backend_suites", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")

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
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        ":constants",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "comparators",
    srcs = ["comparators.cc"],
    hdrs = [
        "comparators.h",
    ],
    deps = [
        ":constants",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "comparators_test",
    srcs = ["comparators_test.cc"],
    deps = [
        ":comparators",
        ":constants",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "constants",
    srcs = ["constants.cc"],
    hdrs = ["constants.h"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "broadcast",
    srcs = ["broadcast.cc"],
    hdrs = ["broadcast.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "constants_test",
    srcs = ["constants_test.cc"],
    deps = [
        ":constants",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "conv_grad_size_util",
    srcs = ["conv_grad_size_util.cc"],
    hdrs = ["conv_grad_size_util.h"],
    deps = [
        "//xla:status_macros",
        "//xla/client:padding",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "dynamic_shaped_ops",
    srcs = ["dynamic_shaped_ops.cc"],
    hdrs = ["dynamic_shaped_ops.h"],
    deps = [
        ":constants",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:value_inference",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "loops",
    srcs = ["loops.cc"],
    hdrs = ["loops.h"],
    deps = [
        ":constants",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "math",
    srcs = ["math.cc"],
    hdrs = ["math.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":loops",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/client:xla_builder",
    ],
)

xla_test(
    name = "math_test",
    srcs = ["math_test.cc"],
    backend_tags = {
        # Times out.
        "ghostfish_iss": ["noasan"],
    },
    deps = [
        ":constants",
        ":math",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "matrix",
    srcs = ["matrix.cc"],
    hdrs = ["matrix.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":slicing",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    deps = [
        ":constants",
        ":matrix",
        ":slicing",
        "//xla:status",
        "//xla:statusor",
        "//xla:test",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":conv_grad_size_util",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

xla_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    deps = [
        ":pooling",
        "//xla:test",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "prng",
    srcs = ["prng.cc"],
    hdrs = ["prng.h"],
    deps = [
        ":constants",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
    ],
)

xla_test(
    name = "prng_test",
    srcs = ["prng_test.cc"],
    deps = [
        ":constants",
        ":prng",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "qr",
    srcs = ["qr.cc"],
    hdrs = ["qr.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":slicing",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "qr_test",
    srcs = ["qr_test.cc"],
    tags = ["optonly"],
    deps = [
        ":constants",
        ":matrix",
        ":qr",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:literal",
        "//xla:statusor",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "lu_decomposition",
    srcs = ["lu_decomposition.cc"],
    hdrs = ["lu_decomposition.h"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
    ],
)

cc_library(
    name = "approx_topk",
    srcs = ["approx_topk.cc"],
    hdrs = ["approx_topk.h"],
    deps = [
        ":approx_topk_shape",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "approx_topk_shape",
    srcs = ["approx_topk_shape.cc"],
    hdrs = ["approx_topk_shape.h"],
    deps = [
        "//xla:statusor",
        "//xla:util",
    ],
)

cc_library(
    name = "slicing",
    srcs = ["slicing.cc"],
    hdrs = ["slicing.h"],
    deps = [
        ":arithmetic",
        ":constants",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "slicing_test",
    srcs = ["slicing_test.cc"],
    deps = [
        ":slicing",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "sorting",
    srcs = ["sorting.cc"],
    hdrs = ["sorting.h"],
    deps = [
        ":comparators",
        ":constants",
        ":loops",
        ":slicing",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
    ],
)

xla_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    deps = [
        ":sorting",
        "//xla:test",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "quantize",
    hdrs = ["quantize.h"],
    deps = [
        ":constants",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@local_tsl//tsl/platform:bfloat16",
    ],
)

xla_test(
    name = "quantize_test",
    srcs = ["quantize_test.cc"],
    # TODO(b/122119490): re-enable TAP after fixing.
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":quantize",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "testing",
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:global_data",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "self_adjoint_eig",
    srcs = ["self_adjoint_eig.cc"],
    hdrs = ["self_adjoint_eig.h"],
    deps = [
        ":arithmetic",
        ":comparators",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":slicing",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "self_adjoint_eig_test",
    srcs = ["self_adjoint_eig_test.cc"],
    real_hardware_only = True,
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        ":arithmetic",
        ":constants",
        ":math",
        ":matrix",
        ":self_adjoint_eig",
        "//xla:array",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:literal",
        "//xla:statusor",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "svd",
    srcs = ["svd.cc"],
    hdrs = ["svd.h"],
    deps = [
        ":arithmetic",
        ":comparators",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":slicing",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "svd_test",
    srcs = ["svd_test.cc"],
    real_hardware_only = True,
    shard_count = 10,
    tags = ["optonly"],
    deps = [
        ":arithmetic",
        ":constants",
        ":matrix",
        ":slicing",
        ":svd",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "tridiagonal",
    srcs = ["tridiagonal.cc"],
    hdrs = ["tridiagonal.h"],
    deps = [
        ":constants",
        ":loops",
        ":slicing",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "tridiagonal_test",
    srcs = ["tridiagonal_test.cc"],
    real_hardware_only = True,
    shard_count = 10,
    tags = ["optonly"],
    deps = [
        ":slicing",
        ":tridiagonal",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "logdet",
    srcs = ["logdet.cc"],
    hdrs = ["logdet.h"],
    deps = [
        ":arithmetic",
        ":constants",
        ":loops",
        ":math",
        ":matrix",
        ":qr",
        ":slicing",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/client:xla_builder",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_test(
    name = "logdet_test",
    srcs = ["logdet_test.cc"],
    tags = [
        "optonly",
    ],
    deps = [
        ":logdet",
        ":matrix",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:literal",
        "//xla:statusor",
        "//xla:test",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "tuple",
    srcs = ["tuple.cc"],
    hdrs = ["tuple.h"],
    deps = [
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/client:xla_builder",
        "@com_google_absl//absl/container:inlined_vector",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "tuple_test",
    srcs = ["tuple_test.cc"],
    deps = [
        ":tuple",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:xla_builder",
        "//xla/tests:client_library_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)
