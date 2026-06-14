load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "@xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],
)

# TODO(bmzhao): This target a holdover from tensorflow/core/BUILD. We
# will add proper dependencies once tf/core/graph/BUILD has granular
# targets added in a subsequent changes.
cc_library(
    name = "mkl_graph_util",
    hdrs = ["mkl_graph_util.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "zen_graph_util",
    hdrs = ["zen_graph_util.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "algorithm",
    srcs = ["algorithm.cc"],
    hdrs = ["algorithm.h"],
    deps = [
        ":graph_cluster",
        "//tensorflow/core:framework_internal_headers_lib_gather",
        "//tensorflow/core:portable_gif_internal",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "collective_order",
    srcs = ["collective_order.cc"],
    hdrs = ["collective_order.h"],
    deps = [
        ":algorithm",
        ":graph_cluster",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "control_flow",
    srcs = ["control_flow.cc"],
    hdrs = ["control_flow.h"],
    deps = [
        ":graph_cluster",
        ":node_builder",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "costmodel",
    srcs = ["costmodel.cc"],
    hdrs = ["costmodel.h"],
    deps = [
        ":graph_cluster",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core:framework_internal_headers_lib_gather",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:allocation_description_proto_cc",
        "//tensorflow/core/framework:cost_graph_proto_cc",
        "//tensorflow/core/framework:step_stats_proto_cc",
        "//tensorflow/core/framework:tensor_description_proto_cc",
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:protobuf",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "edgeset",
    srcs = ["edgeset.cc"],
    hdrs = ["edgeset.h"],
    deps = [
        "//tensorflow/core:framework_internal_headers_lib_gather",
        "//tensorflow/core:portable_gif_internal",
    ],
)

cc_library(
    name = "graph_debug_info_builder",
    srcs = ["graph_debug_info_builder.cc"],
    hdrs = ["graph_debug_info_builder.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/config:flag_defs",
        "//tensorflow/core/framework:graph_debug_info_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:stack_frame",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "graph_def_builder",
    srcs = ["graph_def_builder.cc"],
    hdrs = ["graph_def_builder.h"],
    deps = [
        ":graph_cluster",
        ":node_builder",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core:framework_internal_headers_lib_gather",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "graph_partition",
    srcs = ["graph_partition.cc"],
    hdrs = ["graph_partition.h"],
    deps = [
        ":algorithm",
        ":control_flow",
        ":costmodel",
        ":graph_cluster",
        ":graph_debug_info_builder",
        ":graph_def_builder",
        ":node_builder",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "node_builder",
    srcs = ["node_builder.cc"],
    hdrs = ["node_builder.h"],
    deps = [
        ":graph_cluster",
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal_headers_lib_gather",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "optimizer_cse",
    srcs = ["optimizer_cse.cc"],
    hdrs = ["optimizer_cse.h"],
    deps = [
        ":algorithm",
        ":graph_cluster",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:protobuf",
        "//third_party/protobuf:protobuf_lite",
        "//third_party/protobuf/io",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "graph_cluster",
    srcs = [
        "graph.cc",
        "graph_node_util.cc",
    ],
    hdrs = [
        "graph.h",
        "graph_node_util.h",
    ],
    deps = [
        ":edgeset",
        ":graph_debug_info_builder",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal_headers_lib_gather",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:portable_jpeg_internal",
        "//tensorflow/core/framework:full_type_proto_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_debug_info_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:node_properties",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:op_def_builder",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/lib/core:arena",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/public:version",
        "//third_party/protobuf:protobuf_lite",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "collective_order_test",
    size = "small",
    srcs = [
        "collective_order_test.cc",
    ],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "core_cpu_headers",
    srcs = [
        "algorithm.h",
        "default_device.h",
        "graph.h",
        "graph_debug_info_builder.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "node_builder.h",
        "validate.h",
        "while_context.h",
    ],
)

filegroup(
    name = "framework_internal_private_headers",
    srcs = [
        "edgeset.h",
        "graph.h",
        "graph_debug_info_builder.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "node_builder.h",
        "tensor_id.h",
        "types.h",
    ],
)

filegroup(
    name = "framework_internal_impl_srcs",
    srcs = [
        "edgeset.cc",
        "graph.cc",
        "graph_debug_info_builder.cc",
        "graph_def_builder.cc",
        "graph_node_util.cc",
        "node_builder.cc",
        "tensor_id.cc",
        "while_context.cc",
        "while_context.h",
    ],
)

# Note(bmzhao): This target is a holdover from the GRAPH_HDRS array
# in tensorflow/core/BUILD. This target contains all '.h' files under
# tensorflow/core/graph, except for the following:
# 'benchmark_testlib.h'.
filegroup(
    name = "graph_headers",
    srcs = [
        "algorithm.h",
        "collective_order.h",
        "colors.h",
        "control_flow.h",
        "costmodel.h",
        "default_device.h",
        "edgeset.h",
        "graph.h",
        "graph_debug_info_builder.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "graph_partition.h",
        "node_builder.h",
        "optimizer_cse.h",
        "subgraph.h",
        "tensor_id.h",
        "testlib.h",
        "types.h",
        "validate.h",
        "while_context.h",
    ],
)

filegroup(
    name = "graph_srcs",
    srcs = [
        "algorithm.cc",
        "collective_order.cc",
        "colors.cc",
        "control_flow.cc",
        "costmodel.cc",
        "graph_partition.cc",
        "optimizer_cse.cc",
        "subgraph.cc",
        "validate.cc",
    ],
)

filegroup(
    name = "testlib_headers",
    srcs = [
        "benchmark_testlib.h",
        "testlib.h",
    ] + if_mkl(["mkl_testlib.h"]),
)

filegroup(
    name = "testlib_srcs",
    srcs = [
        "testlib.cc",
    ] + if_mkl(["mkl_testlib.cc"]),
)

filegroup(
    name = "mkl_graph_util_header",
    srcs = [
        "mkl_graph_util.h",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "algorithm.cc",
        "collective_order.cc",
        "colors.cc",
        "control_flow.cc",
        "costmodel.cc",
        "edgeset.cc",
        "graph.cc",
        "graph_debug_info_builder.cc",
        "graph_def_builder.cc",
        "graph_node_util.cc",
        "graph_partition.cc",
        "node_builder.cc",
        "optimizer_cse.cc",
        "subgraph.cc",
        "tensor_id.cc",
        "validate.cc",
        "while_context.cc",
    ],
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "algorithm.h",
        "benchmark_testlib.h",
        "collective_order.h",
        "colors.h",
        "control_flow.h",
        "costmodel.h",
        "default_device.h",
        "edgeset.h",
        "graph.h",
        "graph_debug_info_builder.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "graph_partition.h",
        "mkl_graph_util.h",
        "node_builder.h",
        "optimizer_cse.h",
        "subgraph.h",
        "tensor_id.h",
        "testlib.h",
        "types.h",
        "validate.h",
        "while_context.h",
    ],
)

# Note(bmzhao): Ideally we would use a filegroup to represent these tests instead.
# However, that causes tf_cc_tests to link all of these tests into a single object
# file. This breaks tensorflow/core:core_higher_level_tests, because some of these
# tests redefine the same symbol. This will be fixed by having granular tests
# instead, after phase 4 of the tensorflow's build refactoring:
# https://github.com/tensorflow/community/pull/179
exports_files(
    srcs = [
        "algorithm_test.cc",
        "control_flow_test.cc",
        "costmodel_test.cc",
        "edgeset_test.cc",
        "graph_debug_info_builder_test.cc",
        "graph_def_builder_test.cc",
        "graph_partition_test.cc",
        "graph_test.cc",
        "node_builder_test.cc",
        "optimizer_cse_test.cc",
        "subgraph_test.cc",
        "tensor_id_test.cc",
        "validate_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

tf_cc_tests(
    name = "higher_level_tests",
    size = "small",
    srcs = [
        "algorithm_test.cc",
        "control_flow_test.cc",
        "costmodel_test.cc",
        "edgeset_test.cc",
        "graph_debug_info_builder_test.cc",
        "graph_def_builder_test.cc",
        "graph_partition_test.cc",
        "graph_test.cc",
        "node_builder_test.cc",
        "optimizer_cse_test.cc",
        "subgraph_test.cc",
        "tensor_id_test.cc",
        "validate_test.cc",
    ],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/util:protos_test_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)
