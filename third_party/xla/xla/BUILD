load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "@local_tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//third_party/compute_library:build_defs.bzl", "if_enable_acl")

# Placeholder: load py_proto_library
load("//xla:xla.bzl", "xla_cc_test", "xla_py_proto_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:internal"],
    packages = [
        # copybara:uncomment "//learning/...",
        "//third_party/australis/...",
        "//third_party/iree/...",
        "//third_party/libxc/...",
        "//third_party/mira/...",
        "//third_party/mlcompass/...",
        "//third_party/mlir_edge/model_curriculum/...",
        "//third_party/openxla/shardonnay/...",
        "//third_party/py/enzyme_ad/...",
        "//third_party/py/jax/...",
        "//third_party/py/t5x/...",
        "//third_party/py/tpu_graphs/...",
        "//tensorflow/compiler/...",
        "//tensorflow/python/tpu/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "//xla/...",
    ],
)

package_group(
    name = "runtime",
    packages = [
        "//xla/runtime/...",
        "//xla/service/gpu/runtime/...",
    ],
)

exports_files([
    "lit.cfg.py",
])

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

filegroup(
    name = "cpu_runtime_srcs",
    srcs = [
        "cpu_function_runtime.cc",
        "executable_run_options.cc",
    ],
    visibility = internal_visibility([":friends"]),
)

filegroup(
    name = "cpu_runtime_hdrs",
    srcs = [
        "cpu_function_runtime.h",
        "executable_run_options.h",
        "types.h",
    ],
    visibility = internal_visibility([":friends"]),
)

tf_proto_library(
    name = "xla_data_proto",
    srcs = ["xla_data.proto"],
    cc_api_version = 2,
    create_grpc_library = True,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "xla_proto",
    srcs = ["xla.proto"],
    cc_api_version = 2,
    create_grpc_library = True,
    make_default_target_header_only = True,
    protodeps = [
        ":xla_data_proto",
        "//xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bit_cast",
    hdrs = ["bit_cast.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":types",
        "@com_google_absl//absl/base",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:bfloat16",
    ],
)

xla_cc_test(
    name = "bit_cast_test",
    srcs = ["bit_cast_test.cc"],
    deps = [
        ":bit_cast",
        ":test",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:bfloat16",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "comparison_util",
    srcs = [
        "comparison_util.cc",
    ],
    hdrs = [
        "comparison_util.h",
        "primitive_util.h",
    ],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":shape_util",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_cc_test(
    name = "comparison_util_test",
    srcs = ["comparison_util_test.cc"],
    deps = [
        ":comparison_util",
        ":test",
        ":types",
        ":xla_data_proto_cc",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "compiler_macros",
    hdrs = ["compiler_macros.h"],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "ef57",
    srcs = ["ef57.cc"],
    hdrs = ["ef57.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":compiler_macros",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "ef57_test",
    srcs = ["ef57_test.cc"],
    deps = [
        ":ef57",
        ":test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:log_streamer",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "execution_options_util",
    srcs = [
        "execution_options_util.cc",
    ],
    hdrs = [
        "execution_options_util.h",
    ],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":debug_options_flags",
        ":xla_proto_cc",
    ],
)

cc_library(
    name = "frontend_attributes",
    srcs = [
        "frontend_attributes.cc",
    ],
    hdrs = [
        "frontend_attributes.h",
    ],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "test",
    testonly = 1,
    hdrs = ["test.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_cc_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.cc"],
    visibility = ["//visibility:private"],
    deps = [
        ":test",
        ":types",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "service_interface",
    srcs = [],
    hdrs = ["service_interface.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":xla_data_proto_cc",
        ":xla_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "status_macros",
    srcs = ["status_macros.cc"],
    hdrs = ["status_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "status_macros_test",
    size = "small",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":status_macros",
        ":test",
        ":test_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "statusor",
    hdrs = [
        "statusor.h",
    ],
    linkopts = select({
        "//xla/tsl:freebsd": ["-lexecinfo"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = [
        "iterator_util.h",
        "map_util.h",
        "maybe_owning.h",
        "overflow_util.h",
        "util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        ":status_macros",
        ":statusor",
        ":types",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/lib/gtl:iterator_range",
        "@local_tsl//tsl/lib/math:math_util",
        "@local_tsl//tsl/platform:bfloat16",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
    ],
)

xla_cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":test",
        ":types",
        ":util",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "permutation_util",
    srcs = ["permutation_util.cc"],
    hdrs = ["permutation_util.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "permutation_util_test",
    srcs = ["permutation_util_test.cc"],
    deps = [
        ":permutation_util",
        ":test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "protobuf_util",
    srcs = ["protobuf_util.cc"],
    hdrs = [
        "protobuf_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "iterator_util_test",
    srcs = ["iterator_util_test.cc"],
    deps = [
        ":test",
        ":util",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "shape_util",
    srcs = [
        "index_util.cc",
        "layout.cc",
        "layout_util.cc",
        "primitive_util.cc",
        "shape.cc",
        "shape_util.cc",
    ],
    hdrs = [
        "index_util.h",
        "layout.h",
        "layout_util.h",
        "primitive_util.h",
        "shape.h",
        "shape_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":permutation_util",
        ":printer",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "sharding_op_util",
    srcs = ["sharding_op_util.cc"],
    hdrs = ["sharding_op_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status_macros",
        ":util",
        "//xla/service:hlo_lexer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/hash:hash_testing",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "shape_util_test",
    srcs = ["shape_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "primitive_util_test",
    srcs = ["primitive_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":test_helpers",
        ":xla_data_proto_cc",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "layout_util_test",
    srcs = ["layout_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":test_helpers",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "layout_test",
    srcs = ["layout_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "index_util_test",
    srcs = ["index_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "literal",
    srcs = ["literal.cc"],
    hdrs = ["literal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":array2d",
        ":array3d",
        ":array4d",
        ":permutation_util",
        ":printer",
        ":shape_tree",
        ":shape_util",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/tsl/util:byte_swap_array",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/lib/core:bitmap",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "literal_test",
    srcs = ["literal_test.cc"],
    deps = [
        ":array",
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal",
        ":literal_util",
        ":shape_tree",
        ":shape_util",
        ":test",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "literal_util",
    srcs = ["literal_util.cc"],
    hdrs = ["literal_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/core:bitmap",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "error_spec",
    hdrs = ["error_spec.h"],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "literal_comparison",
    srcs = ["literal_comparison.cc"],
    hdrs = ["literal_comparison.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":error_spec",
        ":literal",
        ":literal_util",
        ":shape_util",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "metric_table_report",
    srcs = ["metric_table_report.cc"],
    hdrs = ["metric_table_report.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "device_util",
    hdrs = ["device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":types",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "array_test",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "array2d",
    hdrs = ["array2d.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":types",
        ":util",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "array2d_test",
    srcs = ["array2d_test.cc"],
    deps = [
        ":array2d",
        ":test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "array3d",
    hdrs = ["array3d.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":array",
        ":types",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "array3d_test",
    srcs = ["array3d_test.cc"],
    deps = [
        ":array3d",
        ":test",
        ":types",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "array4d",
    hdrs = ["array4d.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":array",
        ":array2d",
        ":types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "array4d_test",
    srcs = ["array4d_test.cc"],
    deps = [
        ":array4d",
        ":test",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "executable_run_options",
    srcs = ["executable_run_options.cc"],
    hdrs = ["executable_run_options.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "packed_literal_reader",
    srcs = ["packed_literal_reader.cc"],
    hdrs = ["packed_literal_reader.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/base",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "test_helpers",
    testonly = 1,
    hdrs = ["test_helpers.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":statusor",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "text_literal_reader",
    srcs = ["text_literal_reader.cc"],
    hdrs = ["text_literal_reader.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/service:hlo_parser",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/io:buffered_inputstream",
        "@local_tsl//tsl/lib/io:random_inputstream",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "text_literal_reader_test",
    srcs = ["text_literal_reader_test.cc"],
    deps = [
        ":literal",
        ":shape_util",
        ":test",
        ":text_literal_reader",
        ":types",
        ":xla_data_proto_cc",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "text_literal_writer",
    srcs = ["text_literal_writer.cc"],
    hdrs = ["text_literal_writer.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":types",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "text_literal_writer_test",
    srcs = ["text_literal_writer_test.cc"],
    deps = [
        ":literal",
        ":literal_util",
        ":test",
        ":test_helpers",
        ":text_literal_writer",
        ":types",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "shape_tree",
    srcs = ["shape_tree.cc"],
    hdrs = ["shape_tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        ":statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/gtl:iterator_range",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "shape_tree_test",
    srcs = ["shape_tree_test.cc"],
    deps = [
        ":shape_tree",
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "shape_layout",
    srcs = ["shape_layout.cc"],
    hdrs = ["shape_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":printer",
        ":shape_util",
        ":util",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "window_util",
    srcs = ["window_util.cc"],
    hdrs = ["window_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "window_util_test",
    srcs = ["window_util_test.cc"],
    deps = [
        ":test",
        ":window_util",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "reference_util",
    srcs = ["reference_util.cc"],
    hdrs = ["reference_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal_util",
        ":util",
        ":window_util",
        ":xla_data_proto_cc",
        "//xla/client:padding",
        "//xla/client:xla_builder",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/math:math_util",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "reference_util_test",
    srcs = ["reference_util_test.cc"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal",
        ":reference_util",
        ":test",
        ":xla_data_proto_cc",
        "//xla/client:padding",
        "//xla/tests:literal_test_util",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "parse_flags_from_env",
    srcs = ["parse_flags_from_env.cc"],
    hdrs = ["parse_flags_from_env.h"],
    deps =
        [
            ":types",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/types:span",
            "@local_tsl//tsl/platform:logging",
        ],
)

xla_cc_test(
    name = "parse_flags_from_env_test",
    srcs = ["parse_flags_from_env_test.cc"],
    deps =
        [
            ":parse_flags_from_env",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/strings:str_format",
            "@local_tsl//tsl/platform:env",
            "@local_tsl//tsl/platform:logging",
            "@local_tsl//tsl/platform:subprocess",
            "@local_tsl//tsl/platform:test",
        ],
)

cc_library(
    name = "debug_options_flags",
    srcs = [
        "debug_options_flags.cc",
        "debug_options_parsers.h",
    ],
    hdrs = ["debug_options_flags.h"],
    copts = if_enable_acl(["-DXLA_CPU_USE_ACL=1"]),
    visibility = internal_visibility([":friends"]),
    deps =
        [
            ":parse_flags_from_env",
            ":xla_proto_cc",
            "//xla/stream_executor/cuda:ptx_compiler_support",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:node_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@local_tsl//tsl/platform:logging",
            "@local_tsl//tsl/platform:protobuf",
        ],
)

cc_library(
    name = "cpu_function_runtime",
    srcs = ["cpu_function_runtime.cc"],
    hdrs = ["cpu_function_runtime.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

xla_cc_test(
    name = "debug_options_parsers_test",
    size = "small",
    srcs = [
        "debug_options_parsers.h",
        "debug_options_parsers_test.cc",
    ],
    deps =
        [
            ":xla_proto_cc",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/strings",
            "@local_tsl//tsl/platform:test",
        ],
)

cc_library(
    name = "refcounting_hash_map",
    hdrs = ["refcounting_hash_map.h"],
    deps = [
        ":statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "refcounting_hash_map_test",
    srcs = ["refcounting_hash_map_test.cc"],
    deps = [
        ":refcounting_hash_map",
        ":test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
)

cc_library(
    name = "side_effect_util",
    srcs = ["side_effect_util.cc"],
    hdrs = ["side_effect_util.h"],
)

cc_library(
    name = "lazy",
    hdrs = ["lazy.h"],
    deps = ["@com_google_absl//absl/functional:any_invocable"],
)

tf_proto_library(
    name = "autotune_results_proto",
    srcs = ["autotune_results.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [":autotuning_proto"],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "autotune_results_py_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [
        ":autotune_results_proto",
    ],
)

tf_proto_library(
    name = "autotuning_proto",
    srcs = ["autotuning.proto"],
    make_default_target_header_only = True,
    protodeps = ["@local_tsl//tsl/protobuf:dnn_proto"],
)

cc_library(
    name = "printer",
    srcs = ["printer.cc"],
    hdrs = ["printer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@local_tsl//tsl/platform:logging",
    ],
)

# -----------------------------------------------------------------------------

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xla_data_proto_py_pb2",
#     api_version = 2,
#     visibility = internal_visibility([":friends"]),
#     deps = [":xla_data_proto"],
# )
#
# py_proto_library(
#     name = "xla_py_pb2",
#     testonly = 0,
#     api_version = 2,
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = internal_visibility([":friends"]),
#     deps = [":xla_proto"],
# )
# copybara:uncomment_end

cc_library(
    name = "empty",
    visibility = ["//visibility:public"],
)

# Needed to workaround https://github.com/bazelbuild/bazel/issues/21519
alias(
    name = "bazel_issue_21519",
    actual = ":empty",
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "lit_bzl",
    srcs = ["lit.bzl"],
    deps = ["@bazel_skylib//lib:paths"],
)

bzl_library(
    name = "xla_bzl",
    srcs = ["xla.bzl"],
    deps = [
        "//xla/tsl:tsl_bzl",
        "@local_config_rocm//rocm:build_defs_bzl",
        "@local_tsl//tsl/platform:build_config_root_bzl",
        "@local_tsl//tsl/platform/default:cuda_build_defs_bzl",
    ],
)
