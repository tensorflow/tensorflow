load("//tensorflow:tensorflow.bzl", "cc_header_only_library", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//tensorflow:internal"],
    packages = [
        "//tensorflow/compiler/...",
        "//tensorflow/python/tpu/...",
        "//third_party/py/jax/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/xla/...",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

tf_proto_library_cc(
    name = "xla_data_proto",
    srcs = ["xla_data.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "xla_data_proto",  # bzl adds a _py suffix
    srcs = ["xla_data.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "xla_proto",
    srcs = ["xla.proto"],
    cc_api_version = 2,
    protodeps = [
        ":xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "xla_proto",  # bzl adds a _py suffix
    srcs = ["xla.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_data_proto_py",
        "//tensorflow/compiler/xla/service:hlo_proto_py",
    ],
)

cc_library(
    name = "bit_cast",
    hdrs = ["bit_cast.h"],
    visibility = [":friends"],
    deps = [
        ":types",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
    ],
)

tf_cc_test(
    name = "bit_cast_test",
    srcs = ["bit_cast_test.cc"],
    deps = [
        ":bit_cast",
        ":test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "comparison_util",
    srcs = [
        "comparison_util.cc",
    ],
    hdrs = [
        "comparison_util.h",
    ],
    visibility = [":friends"],
    deps = [
        ":statusor",
        ":types",
        ":util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
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
    name = "test",
    testonly = 1,
    hdrs = ["test.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "service_interface",
    srcs = [],
    hdrs = ["service_interface.h"],
    visibility = [":friends"],
    deps = [
        ":status",
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
        ":statusor",
        ":types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "status_macros_test",
    size = "small",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":status_macros",
        ":statusor",
        ":test",
        ":test_helpers",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "statusor",
    hdrs = [
        "statusor.h",
    ],
    linkopts = select({
        "//tensorflow:freebsd": ["-lexecinfo"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        "//tensorflow/stream_executor/lib",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:numbers",
        "//third_party/eigen3",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":test",
        ":types",
        ":util",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "iterator_util_test",
    srcs = ["iterator_util_test.cc"],
    deps = [
        ":test",
        ":util",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
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
        ":protobuf_util",
        ":status",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:regexp_internal",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "shape_util_test",
    srcs = ["shape_util_test.cc"],
    deps = [
        ":shape_util",
        ":status_macros",
        ":test",
        ":test_helpers",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "layout_util_test",
    srcs = ["layout_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":test_helpers",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "layout_test",
    srcs = ["layout_test.cc"],
    deps = [
        ":shape_util",
        ":status_macros",
        ":test",
        ":test_helpers",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "index_util_test",
    srcs = ["index_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "//tensorflow/core:test_main",
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
        ":shape_util",
        ":sparse_index_array",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "literal_test",
    srcs = ["literal_test.cc"],
    deps = [
        ":array3d",
        ":array4d",
        ":literal",
        ":literal_util",
        ":shape_util",
        ":test",
        ":types",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
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
        ":sparse_index_array",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "error_spec",
    hdrs = ["error_spec.h"],
)

cc_library(
    name = "literal_comparison",
    srcs = ["literal_comparison.cc"],
    hdrs = ["literal_comparison.h"],
    deps = [
        ":error_spec",
        ":literal",
        ":literal_util",
        ":util",
        "//tensorflow/core:lib",
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
        ":util",
        "//tensorflow/core:lib",
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
        "//tensorflow/core:stream_executor_no_cuda",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    deps = [
        ":status",
        ":types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "array_test",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "array2d",
    hdrs = ["array2d.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":array",
        ":types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "array2d_test",
    srcs = ["array2d_test.cc"],
    deps = [
        ":array2d",
        ":test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "array3d",
    hdrs = ["array3d.h"],
    visibility = [":friends"],
    deps = [
        ":array",
        ":types",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "array3d_test",
    srcs = ["array3d_test.cc"],
    deps = [
        ":array3d",
        ":test",
        ":types",
        "//tensorflow/core:test_main",
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "array4d_test",
    srcs = ["array4d_test.cc"],
    deps = [
        ":array4d",
        ":test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable_run_options",
    srcs = ["executable_run_options.cc"],
    hdrs = ["executable_run_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "@com_google_absl//absl/strings",
    ],
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core:test",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "text_literal_reader_test",
    srcs = ["text_literal_reader_test.cc"],
    deps = [
        ":literal",
        ":shape_util",
        ":test",
        ":text_literal_reader",
        ":types",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "text_literal_writer_test",
    srcs = ["text_literal_writer_test.cc"],
    deps = [
        ":literal",
        ":literal_util",
        ":test",
        ":test_helpers",
        ":text_literal_writer",
        ":types",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "shape_tree",
    hdrs = ["shape_tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        ":status_macros",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "shape_tree_test",
    srcs = ["shape_tree_test.cc"],
    deps = [
        ":shape_tree",
        ":shape_util",
        ":test",
        ":xla_data_proto_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "shape_layout",
    srcs = ["shape_layout.cc"],
    hdrs = ["shape_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape_util",
        ":types",
        ":util",
        ":xla_data_proto_cc",
        "//tensorflow/core:lib",
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "window_util_test",
    srcs = ["window_util_test.cc"],
    deps = [
        ":test",
        ":window_util",
        "//tensorflow/core:test_main",
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
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "sparse_index_array",
    srcs = ["sparse_index_array.cc"],
    hdrs = ["sparse_index_array.h"],
    deps = [
        ":array2d",
        ":shape_util",
        ":xla_data_proto_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "sparse_index_array_test",
    srcs = ["sparse_index_array_test.cc"],
    deps = [
        ":sparse_index_array",
        ":test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "parse_flags_from_env",
    srcs = ["parse_flags_from_env.cc"],
    hdrs = ["parse_flags_from_env.h"],
    deps =
        [
            ":types",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/types:span",
        ],
)

tf_cc_test(
    name = "parse_flags_from_env_test",
    srcs = ["parse_flags_from_env_test.cc"],
    deps =
        [
            ":parse_flags_from_env",
            ":types",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
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
    visibility = [":friends"],
    deps =
        [
            ":parse_flags_from_env",
            ":xla_proto_cc",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
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
    visibility = [":friends"],
    deps = ["//tensorflow/core:framework_lite"],
)

tf_cc_test(
    name = "debug_options_parsers_test",
    size = "small",
    srcs = [
        "debug_options_parsers.h",
        "debug_options_parsers_test.cc",
    ],
    deps =
        [
            ":xla_proto_cc",
            "//tensorflow/compiler/xla/service:hlo",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
        ],
)

cc_library(
    name = "refcounting_hash_map",
    hdrs = ["refcounting_hash_map.h"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "refcounting_hash_map_test",
    srcs = ["refcounting_hash_map_test.cc"],
    deps = [
        ":refcounting_hash_map",
        ":test",
        "//tensorflow/core:test_main",
    ],
)

# -----------------------------------------------------------------------------

# This is a headers target that extra XLA devices can use to prevent circular dependencies.  Devices that are compiled as separate shared objects can also use it to prevent linking of library code.
cc_header_only_library(
    name = "xla_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_data_proto_cc",
        ":xla_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:stream_executor_headers_lib",
    ],
)
