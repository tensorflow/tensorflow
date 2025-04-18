load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//third_party/compute_library:build_defs.bzl", "if_enable_acl")

# copybara:uncomment load("@rules_python//python:proto.bzl", "py_proto_library")
load("//xla:package_groups.bzl", "xla_package_groups")
load("//xla:xla.default.bzl", "xla_bzl_library", "xla_cc_test", "xla_py_proto_library")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
    # copybara:uncomment "tf_pyclif_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

exports_files([
    "lit.cfg.py",
] + if_google(["lit_google_cfg.py"]))

xla_package_groups()

bzl_library(
    name = "package_groups_bzl",
    srcs = ["package_groups.bzl"],
    visibility = ["//visibility:public"],
)

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
    create_grpc_library = True,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "xla_proto",
    srcs = ["xla.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
    protodeps = [
        ":xla_data_proto",
        "//xla/service:hlo_proto",
    ] + if_google(["@com_google_protobuf//:any"]),
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
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:bfloat16",
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
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

xla_cc_test(
    name = "comparison_util_test",
    srcs = ["comparison_util_test.cc"],
    deps = [
        ":comparison_util",
        ":types",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "ef57_test",
    srcs = ["ef57_test.cc"],
    deps = [
        ":ef57",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        ":types",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "status_macros",
    srcs = ["status_macros.cc"],
    hdrs = ["status_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:stacktrace",
    ],
)

xla_cc_test(
    name = "status_macros_test",
    size = "small",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":status_macros",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "fp_util",
    hdrs = ["fp_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        ":util",
    ],
)

xla_cc_test(
    name = "fp_util_test",
    srcs = ["fp_util_test.cc"],
    deps = [
        ":bit_cast",
        ":fp_util",
        ":util",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:ml_dtypes",
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
        ":status_macros",
        ":types",
        ":xla_data_proto_cc",
        "//xla/tsl/lib/gtl:iterator_range",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:bfloat16",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:stacktrace",
    ],
)

xla_cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@ml_dtypes_py//ml_dtypes:float8",
    ],
)

cc_library(
    name = "permutation_util",
    srcs = ["permutation_util.cc"],
    hdrs = ["permutation_util.h"],
    deps = [
        ":types",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "permutation_util_test",
    srcs = ["permutation_util_test.cc"],
    deps = [
        ":permutation_util",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "iterator_util_test",
    srcs = ["iterator_util_test.cc"],
    deps = [
        ":util",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        "shape_partition.cc",
        "shape_util.cc",
    ],
    hdrs = [
        "index_util.h",
        "layout.h",
        "layout_util.h",
        "primitive_util.h",
        "shape.h",
        "shape_partition.h",
        "shape_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":permutation_util",
        ":printer",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:platform_port",
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
        "//xla/hlo/parser:hlo_lexer",
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
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "shape_util_test",
    srcs = ["shape_util_test.cc"],
    deps = [
        ":shape_util",
        ":util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "shape_partition_test",
    srcs = ["shape_partition_test.cc"],
    deps = [
        ":shape_util",
        ":util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "primitive_util_test",
    srcs = ["primitive_util_test.cc"],
    deps = [
        ":shape_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "layout_util_test",
    srcs = ["layout_util_test.cc"],
    deps = [
        ":shape_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "layout_test",
    srcs = ["layout_test.cc"],
    deps = [
        ":shape_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "index_util_test",
    srcs = ["index_util_test.cc"],
    deps = [
        ":shape_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
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
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:byte_swap_array",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:platform_port",
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
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "literal_pool",
    srcs = ["literal_pool.cc"],
    hdrs = ["literal_pool.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":literal",
        ":shape_util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "literal_pool_test",
    srcs = ["literal_pool_test.cc"],
    deps = [
        ":literal_pool",
        ":literal_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "error_spec",
    hdrs = ["error_spec.h"],
    visibility = internal_visibility([":friends"]),
    deps = [":xla_data_proto_cc"],
)

xla_cc_test(
    name = "literal_comparison_test",
    srcs = ["literal_comparison_test.cc"],
    deps = [
        ":error_spec",
        ":literal_comparison",
        ":literal_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "literal_comparison",
    srcs = ["literal_comparison.cc"],
    hdrs = ["literal_comparison.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":error_spec",
        ":fp_util",
        ":literal",
        ":literal_util",
        ":shape_util",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "metric_table_report",
    srcs = ["metric_table_report.cc"],
    hdrs = ["metric_table_report.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":types",
        "@com_google_absl//absl/container:inlined_vector",
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
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
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
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "array3d",
    hdrs = ["array3d.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":array",
        ":types",
        "//xla/tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "array3d_test",
    srcs = ["array3d_test.cc"],
    deps = [
        ":array3d",
        ":types",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "array4d_test",
    srcs = ["array4d_test.cc"],
    deps = [
        ":array2d",
        ":array4d",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
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
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/lib/io:buffered_inputstream",
        "//xla/tsl/lib/io:random_inputstream",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "text_literal_reader_test",
    srcs = ["text_literal_reader_test.cc"],
    deps = [
        ":literal",
        ":shape_util",
        ":text_literal_reader",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        ":types",
        ":xla_data_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "text_literal_writer_test",
    srcs = ["text_literal_writer_test.cc"],
    deps = [
        ":literal_util",
        ":text_literal_writer",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "shape_tree",
    srcs = ["shape_tree.cc"],
    hdrs = ["shape_tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        "//xla/tsl/lib/gtl:iterator_range",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "shape_tree_test",
    srcs = ["shape_tree_test.cc"],
    deps = [
        ":shape_tree",
        ":shape_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "window_util_test",
    srcs = ["window_util_test.cc"],
    deps = [
        ":window_util",
        ":xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
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
        ":literal",
        ":literal_util",
        ":shape_util",
        ":util",
        ":window_util",
        ":xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:shape_inference",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "reference_util_test",
    srcs = ["reference_util_test.cc"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":error_spec",
        ":literal",
        ":literal_util",
        ":reference_util",
        ":xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/testlib:test",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parse_flags_from_env",
    srcs = ["parse_flags_from_env.cc"],
    hdrs = ["parse_flags_from_env.h"],
    deps =
        [
            ":types",
            "//xla/tsl/platform:logging",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/log",
            "@com_google_absl//absl/log:check",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/synchronization",
            "@com_google_absl//absl/types:span",
        ],
)

xla_cc_test(
    name = "parse_flags_from_env_test",
    srcs = ["parse_flags_from_env_test.cc"],
    deps =
        [
            ":parse_flags_from_env",
            "//xla/tsl/platform:env",
            "//xla/tsl/platform:logging",
            "//xla/tsl/platform:macros",
            "//xla/tsl/platform:subprocess",
            "//xla/tsl/platform:test",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/log:check",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_googletest//:gtest",
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
            "//xla/service:collective_utils",
            "//xla/stream_executor/cuda:nvjitlink_support",
            "//xla/stream_executor/cuda:ptx_compiler_support",
            "//xla/tsl/platform:logging",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:node_hash_map",
            "@com_google_absl//absl/log",
            "@com_google_absl//absl/log:check",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@local_tsl//tsl/platform",
            "@local_tsl//tsl/platform:platform_port",
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
        "//xla/backends/cpu:alignment",
        "//xla/tsl/util:safe_reinterpret_cast",
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
            ":debug_options_flags",
            ":parse_flags_from_env",
            ":xla_proto_cc",
            "//xla/tsl/platform:env",
            "//xla/tsl/platform:test",
            "//xla/tsl/util:command_line_flags",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_googletest//:gtest",
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
    make_default_target_header_only = True,
    protodeps = [":autotuning_proto"],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "autotune_results_py_pb2",
    visibility = ["//visibility:public"],
    deps = [
        ":autotune_results_proto",
    ],
)

# copybara:uncomment_begin(google-only)
# tf_pyclif_proto_library(
#     name = "autotune_results_pyclif",
#     proto_lib = ":autotune_results_proto",
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "autotuning_proto",
    srcs = ["autotuning.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//xla/tsl/protobuf:dnn_proto",
    ] + if_google([
        "@com_google_protobuf//:any",
        "@com_google_protobuf//:duration",
    ]),
)

cc_library(
    name = "autotune_result_wrapper",
    srcs = ["autotune_result_wrapper.cc"],
    hdrs = ["autotune_result_wrapper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":autotune_results_proto_cc",
        ":autotuning_proto_cc",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "autotune_result_wrapper_test",
    srcs = ["autotune_result_wrapper_test.cc"],
    deps = [
        ":autotune_result_wrapper",
        ":autotune_results_proto_cc",
        ":autotuning_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "printer",
    srcs = ["printer.cc"],
    hdrs = ["printer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:span",
    ],
)

# -----------------------------------------------------------------------------
# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xla_data_proto_py_pb2",
#     visibility = internal_visibility([":friends"]),
#     deps = [":xla_data_proto"],
# )
#
# py_proto_library(
#     name = "xla_py_pb2",
#     testonly = 0,
#     compatible_with = get_compatible_with_portable(),
#     visibility = internal_visibility([":friends"]),
#     deps = [":xla_proto"],
# )
# copybara:uncomment_end

cc_library(
    name = "sort_json",
    srcs = ["sort_json.cc"],
    hdrs = ["sort_json.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "sort_json_test",
    srcs = ["sort_json_test.cc"],
    deps = [
        ":sort_json",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "online_topsort",
    hdrs = ["online_topsort.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "online_topsort_test",
    srcs = ["online_topsort_test.cc"],
    deps = [
        ":online_topsort",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

bzl_library(
    name = "lit_bzl",
    srcs = ["lit.bzl"],
    deps = [
        "//xla/tsl:package_groups_bzl",
        "//xla/tsl:tsl_bzl",
        "//xla/tsl/platform/default:cuda_build_defs_bzl",
        "@bazel_skylib//lib:paths",
    ],
)

xla_bzl_library()
