load("//tensorflow/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//third_party/compute_library:build_defs.bzl", "if_enable_acl")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test", "xla_py_proto_library")
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/xla:internal"],
    packages = [
        # copybara:uncomment "//learning/infra/mira/...",
        "//third_party/australis/...",
        "//third_party/iree/...",
        "//third_party/mira/...",
        "//third_party/mlcompass/...",
        "//third_party/mlir_edge/model_curriculum/...",
        "//third_party/py/jax/...",
        "//third_party/py/t5x/...",
        "//tensorflow/compiler/...",
        "//tensorflow/python/tpu/...",
    ],
)

package_group(
    name = "internal",
    includes = ["//tensorflow:internal"],
    packages = [
        "//tensorflow/compiler/xla/...",
    ],
)

package_group(
    name = "runtime",
    packages = [
        "//tensorflow/compiler/xla/runtime/...",
        "//tensorflow/compiler/xla/service/gpu/runtime/...",
    ],
)

exports_files(["run_lit.sh"])

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
    visibility = [":friends"],
)

filegroup(
    name = "cpu_runtime_hdrs",
    srcs = [
        "cpu_function_runtime.h",
        "executable_run_options.h",
        "types.h",
    ],
    visibility = [":friends"],
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
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bit_cast",
    hdrs = ["bit_cast.h"],
    visibility = [":friends"],
    deps = [
        ":types",
        "//tensorflow/tsl/platform:bfloat16",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
    ],
)

xla_cc_test(
    name = "bit_cast_test",
    srcs = ["bit_cast_test.cc"],
    deps = [
        ":bit_cast",
        ":test",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test_main",
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
    visibility = [":friends"],
    deps = [
        ":shape_util",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:float8",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "comparison_util_test",
    srcs = ["comparison_util_test.cc"],
    deps = [
        ":comparison_util",
        ":test",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:bfloat16",
        "//tensorflow/tsl/platform:test_main",
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
    visibility = [":friends"],
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
    visibility = [":friends"],
    deps = ["//tensorflow/compiler/xla/hlo/ir:hlo"],
)

cc_library(
    name = "test",
    testonly = 1,
    hdrs = ["test.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [":friends"],
    deps = [
        "//third_party/eigen3",
    ],
)

xla_cc_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.cc"],
    visibility = [":friends"],
    deps = [
        ":test",
        ":types",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "service_interface",
    srcs = [],
    hdrs = ["service_interface.h"],
    visibility = [":friends"],
    deps = [
        ":status",
        ":xla_cc_grpc_proto",
        ":xla_data_proto_cc",
        ":xla_proto_cc",
    ],
)

cc_library(
    name = "status_macros",
    srcs = ["status_macros.cc"],
    hdrs = ["status_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        ":statusor",
        ":types",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:stacktrace",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "status_macros_test",
    size = "small",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":status_macros",
        ":statusor",
        ":test",
        ":test_helpers",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "statusor",
    hdrs = [
        "statusor.h",
    ],
    linkopts = select({
        "//tensorflow/tsl:freebsd": ["-lexecinfo"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = [
        "iterator_util.h",
        "map_util.h",
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
        "//tensorflow/tsl/lib/gtl:iterator_range",
        "//tensorflow/tsl/lib/math:math_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:stacktrace",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":test",
        ":types",
        ":util",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "permutation_util",
    srcs = ["permutation_util.cc"],
    hdrs = ["permutation_util.h"],
    deps = [
        ":types",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "permutation_util_test",
    srcs = ["permutation_util_test.cc"],
    deps = [
        ":permutation_util",
        ":test",
        "//tensorflow/tsl/platform:test_main",
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
        ":status",
        ":status_macros",
        ":types",
        ":util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/hash",
    ],
)

xla_cc_test(
    name = "iterator_util_test",
    srcs = ["iterator_util_test.cc"],
    deps = [
        ":test",
        ":util",
        "//tensorflow/tsl/platform:test_main",
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
        ":status",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sharding_op_util",
    srcs = ["sharding_op_util.cc"],
    hdrs = ["sharding_op_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        ":status_macros",
        "//tensorflow/compiler/xla/service:hlo_lexer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":shape_util",
        ":status_macros",
        ":test",
        ":test_helpers",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "shape_util_test",
    srcs = ["shape_util_test.cc"],
    deps = [
        ":permutation_util",
        ":shape_util",
        ":status_macros",
        ":test",
        ":test_helpers",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "primitive_util_test",
    srcs = ["primitive_util_test.cc"],
    deps = [
        ":shape_util",
        ":status_macros",
        ":test",
        ":test_helpers",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "layout_util_test",
    srcs = ["layout_util_test.cc"],
    deps = [
        ":shape_util",
        ":test_helpers",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "layout_test",
    srcs = ["layout_test.cc"],
    deps = [
        ":shape_util",
        ":status",
        ":test",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "index_util_test",
    srcs = ["index_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "literal",
    srcs = ["literal.cc"],
    hdrs = ["literal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":permutation_util",
        ":printer",
        ":shape_util",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/lib/core:bitmap",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/util:byte_swap_array",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "literal_test",
    srcs = ["literal_test.cc"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal",
        ":literal_util",
        ":shape_util",
        ":test",
        ":types",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "literal_util",
    srcs = ["literal_util.cc"],
    hdrs = ["literal_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal",
        ":shape_util",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/lib/core:bitmap",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "error_spec",
    hdrs = ["error_spec.h"],
    visibility = [":friends"],
)

cc_library(
    name = "literal_comparison",
    srcs = ["literal_comparison.cc"],
    hdrs = ["literal_comparison.h"],
    visibility = [":friends"],
    deps = [
        ":error_spec",
        ":literal",
        ":literal_util",
        ":shape_util",
        ":util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "metric_table_report",
    srcs = ["metric_table_report.cc"],
    hdrs = ["metric_table_report.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "device_util",
    hdrs = ["device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//tensorflow/compiler/xla/stream_executor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    visibility = [":friends"],
    deps = [
        ":status",
        ":types",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/functional:function_ref",
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
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "array3d",
    hdrs = ["array3d.h"],
    visibility = [":friends"],
    deps = [
        ":array",
        ":types",
        "//tensorflow/tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "array3d_test",
    srcs = ["array3d_test.cc"],
    deps = [
        ":array3d",
        ":test",
        ":types",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "array4d",
    hdrs = ["array4d.h"],
    visibility = [":friends"],
    deps = [
        ":array",
        ":array2d",
        ":types",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "array4d_test",
    srcs = ["array4d_test.cc"],
    deps = [
        ":array4d",
        ":test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable_run_options",
    srcs = ["executable_run_options.cc"],
    hdrs = ["executable_run_options.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "packed_literal_reader",
    srcs = ["packed_literal_reader.cc"],
    hdrs = ["packed_literal_reader.h"],
    visibility = [":internal"],
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "test_helpers",
    testonly = 1,
    hdrs = ["test_helpers.h"],
    visibility = [":internal"],
    deps = [
        ":statusor",
        ":types",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "text_literal_reader",
    srcs = ["text_literal_reader.cc"],
    hdrs = ["text_literal_reader.h"],
    visibility = [":internal"],
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/tsl/lib/io:buffered_inputstream",
        "//tensorflow/tsl/lib/io:random_inputstream",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "text_literal_writer",
    srcs = ["text_literal_writer.cc"],
    hdrs = ["text_literal_writer.h"],
    visibility = [":internal"],
    deps = [
        ":literal",
        ":shape_util",
        ":status_macros",
        ":types",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "shape_tree",
    srcs = ["shape_tree.cc"],
    hdrs = ["shape_tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        ":status_macros",
        "//tensorflow/tsl/lib/gtl:iterator_range",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
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
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
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
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "window_util_test",
    srcs = ["window_util_test.cc"],
    deps = [
        ":test",
        ":window_util",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/hlo/evaluator:hlo_evaluator",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/tsl/lib/math:math_util",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
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
        ":literal",
        ":reference_util",
        ":test",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "parse_flags_from_env",
    srcs = ["parse_flags_from_env.cc"],
    hdrs = ["parse_flags_from_env.h"],
    deps =
        [
            ":types",
            "//tensorflow/tsl/platform:logging",
            "//tensorflow/tsl/util:command_line_flags",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/types:span",
        ],
)

xla_cc_test(
    name = "parse_flags_from_env_test",
    srcs = ["parse_flags_from_env_test.cc"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/tsl/platform:env",
            "//tensorflow/tsl/platform:logging",
            "//tensorflow/tsl/platform:subprocess",
            "//tensorflow/tsl/platform:test",
            "//tensorflow/tsl/util:command_line_flags",
            "@com_google_absl//absl/strings:str_format",
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
    visibility = [":friends"],
    deps =
        [
            ":parse_flags_from_env",
            ":xla_proto_cc",
            "//tensorflow/tsl/platform:logging",
            "//tensorflow/tsl/util:command_line_flags",
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:node_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
        ],
)

cc_library(
    name = "cpu_function_runtime",
    srcs = ["cpu_function_runtime.cc"],
    hdrs = ["cpu_function_runtime.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [":friends"],
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
            "//tensorflow/compiler/xla/hlo/ir:hlo",
            "//tensorflow/tsl/platform:logging",
            "//tensorflow/tsl/platform:test",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
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
        "//tensorflow/tsl/platform:test_main",
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
    protodeps = [
        "//tensorflow/tsl/protobuf:autotuning_proto",
    ],
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

cc_library(
    name = "autotune_serialize",
    srcs = if_cuda_is_configured(["autotune_serialize.cc"]),
    hdrs = if_cuda_is_configured(["autotune_serialize.h"]),
    # TODO(aminim): There appears to be an in-progress refactoring in TF/XLA to
    # mark rules as compatible_with GCE.  The rules that this depends on are
    # not yet marked as compatible, so this one can't be either (yet).
    compatible_with = [],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":autotune_results_proto_cc",
        ":statusor",
        "//tensorflow/compiler/xla/service/gpu:gemm_algorithm_picker",
        "//tensorflow/compiler/xla/service/gpu:gpu_conv_algorithm_picker",
    ]),
)

cc_library(
    name = "printer",
    srcs = ["printer.cc"],
    hdrs = ["printer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

filegroup(
    name = "litfiles",
    srcs = [
        "runlit.cfg.py",
        "runlit.site.cfg.py",
    ],
    visibility = ["//tensorflow/compiler/xla:__subpackages__"],
)

# -----------------------------------------------------------------------------

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xla_data_proto_py_pb2",
#     api_version = 2,
#     visibility = [":friends"],
#     deps = [":xla_data_proto"],
# )
#
# py_proto_library(
#     name = "xla_py_pb2",
#     testonly = 0,
#     api_version = 2,
#     compatible_with = ["//buildenv/target:gce"],
#     visibility = [":friends"],
#     deps = [":xla_proto"],
# )
# copybara:uncomment_end
