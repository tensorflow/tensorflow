licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/...",
        "//tensorflow/contrib/tpu/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/xla/...",
    ],
)

load("//tensorflow:tensorflow.bzl", "cc_header_only_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_py",
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

xla_proto_library(
    name = "xla_data_proto",
    srcs = ["xla_data.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "xla_data_proto",  # bzl adds a _py suffix
    srcs = ["xla_data.proto"],
    visibility = ["//visibility:public"],
)

xla_proto_library(
    name = "xla_proto",
    srcs = ["xla.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/compiler/xla/service:session_proto",
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
        ":xla_proto",
        "//tensorflow/compiler/xla/legacy_flags:debug_options_flags",
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
        ":xla_proto",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "status_macros",
    srcs = ["status_macros.cc"],
    hdrs = ["status_macros.h"],
    visibility = [":friends"],
    deps = [
        ":statusor",
        ":types",
        "//tensorflow/core:lib",
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
    srcs = ["statusor.cc"],
    hdrs = [
        "statusor.h",
        "statusor_internals.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "statusor_test",
    size = "small",
    srcs = ["statusor_test.cc"],
    deps = [
        ":statusor",
        ":test",
        ":types",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = [
        "iterator_util.h",
        "map_util.h",
        "ptr_util.h",
        "util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        ":status_macros",
        ":statusor",
        ":types",
        ":xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
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
    ],
)

cc_library(
    name = "shape_util",
    srcs = [
        "index_util.cc",
        "layout_util.cc",
        "primitive_util.cc",
        "shape_util.cc",
    ],
    hdrs = [
        "index_util.h",
        "layout_util.h",
        "primitive_util.h",
        "shape_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":protobuf_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:regexp_internal",
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
        ":xla_data_proto",
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
    name = "index_util_test",
    srcs = ["index_util_test.cc"],
    deps = [
        ":shape_util",
        ":test",
        ":xla_data_proto",
        "//tensorflow/core:test_main",
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
        ":shape_tree",
        ":shape_util",
        ":sparse_index_array",
        ":status_macros",
        ":types",
        ":util",
        ":xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "literal_util_test",
    srcs = ["literal_util_test.cc"],
    deps = [
        ":array3d",
        ":array4d",
        ":literal_util",
        ":shape_util",
        ":test",
        ":types",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
    ],
)

cc_library(
    name = "device_util",
    hdrs = ["device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "array",
    hdrs = ["array.h"],
    deps = [
        ":status",
        ":types",
        "//tensorflow/core:lib",
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
        ":util",
        "//tensorflow/core:lib",
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
    ],
)

cc_library(
    name = "executable_run_options",
    srcs = ["executable_run_options.cc"],
    hdrs = ["executable_run_options.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "packed_literal_reader",
    srcs = ["packed_literal_reader.cc"],
    hdrs = ["packed_literal_reader.h"],
    visibility = [":internal"],
    deps = [
        ":literal_util",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto",
        "//tensorflow/core:lib",
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
    ],
)

cc_library(
    name = "text_literal_reader",
    srcs = ["text_literal_reader.cc"],
    hdrs = ["text_literal_reader.h"],
    visibility = [":internal"],
    deps = [
        ":literal_util",
        ":shape_util",
        ":status_macros",
        ":statusor",
        ":types",
        ":util",
        ":xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "text_literal_reader_test",
    srcs = ["text_literal_reader_test.cc"],
    deps = [
        ":literal_util",
        ":shape_util",
        ":test",
        ":text_literal_reader",
        ":types",
        ":xla_data_proto",
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
        ":literal_util",
        ":shape_util",
        ":status_macros",
        ":types",
        ":xla_data_proto",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "text_literal_writer_test",
    srcs = ["text_literal_writer_test.cc"],
    deps = [
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
        ":util",
        ":xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "shape_tree_test",
    srcs = ["shape_tree_test.cc"],
    deps = [
        ":shape_tree",
        ":shape_util",
        ":test",
        ":xla_data_proto",
        "//tensorflow/core:test_main",
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
        ":xla_data_proto",
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
        ":xla_data_proto",
        "//tensorflow/core:lib",
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
        ":util",
        ":window_util",
        ":xla_data_proto",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "reference_util_test",
    srcs = ["reference_util_test.cc"],
    deps = [
        ":array2d",
        ":array3d",
        ":array4d",
        ":literal_util",
        ":reference_util",
        ":test",
        ":util",
        ":xla_data_proto",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "sparse_index_array",
    srcs = ["sparse_index_array.cc"],
    hdrs = ["sparse_index_array.h"],
    deps = [
        ":array2d",
        ":shape_util",
        ":xla_data_proto",
        "//tensorflow/core:lib",
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

# -----------------------------------------------------------------------------

# This is a headers target that extra XLA devices can use to prevent circular dependencies.  Devices that are compiled as separate shared objects can also use it to prevent linking of library code.
cc_header_only_library(
    name = "xla_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_data_proto",
        ":xla_proto",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:stream_executor_headers_lib",
    ],
)
