package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)

tf_proto_library_cc(
    name = "types_proto",
    srcs = ["types.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "toco_flags_proto",
    srcs = ["toco_flags.proto"],
    protodeps = [":types_proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "model_flags_proto",
    srcs = ["model_flags.proto"],
    protodeps = [":types_proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "types_proto",
    srcs = [
        "types.proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "toco_flags_proto",
    srcs = [
        "toco_flags.proto",
    ],
    protodeps = [":types_proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "model_flags_proto",
    srcs = [
        "model_flags.proto",
    ],
    protodeps = [":types_proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorflow_core_cc_protos_all",
    deps = ["//tensorflow/core:protos_all_cc"],
)

cc_library(
    name = "runtime",
    hdrs = [
        "runtime/common.h",
        "runtime/types.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/lite/kernels/internal:reference_base",
        "//tensorflow/contrib/lite/kernels/internal:types",
    ],
)

# :model offers the core data structures representing a model (a.k.a. "graph")
# for tooling purposes (not needed at inference runtime).
# That includes the top-level Model structure, and the lower-level Operator,
# Array, Buffer structures, etc.
cc_library(
    name = "model",
    hdrs = [
        "model.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model_flags_proto_cc",
        ":runtime",
        ":toco_port",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "toco_graphviz_dump_options",
    srcs = [
        "toco_graphviz_dump_options.cc",
    ],
    hdrs = [
        "toco_graphviz_dump_options.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "toco_cmdline_flags",
    srcs = [
        "toco_cmdline_flags.cc",
    ],
    hdrs = [
        "toco_cmdline_flags.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model_cmdline_flags",
        ":toco_flags_proto_cc",
        ":toco_port",
        ":types_proto_cc",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "model_cmdline_flags",
    srcs = [
        "model_cmdline_flags.cc",
    ],
    hdrs = [
        "args.h",
        "model_cmdline_flags.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model_flags_proto_cc",
        ":toco_graphviz_dump_options",
        ":toco_port",
        ":types_proto_cc",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "toco_port",
    srcs = [
        "toco_port.cc",
    ],
    hdrs = [
        "format_port.h",
        "toco_port.h",
        "toco_types.h",
    ],
    deps = [
        # Placeholder for internal file dependency.
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "graph_transformations",
    srcs = [
        "graph_transformations/convert_expanddims_to_reshape.cc",
        "graph_transformations/convert_pure_conv_to_depthwise.cc",
        "graph_transformations/convert_trivial_transpose_to_reshape.cc",
        "graph_transformations/create_im2col_arrays.cc",
        "graph_transformations/dequantize.cc",
        "graph_transformations/drop_fake_quant.cc",
        "graph_transformations/drop_im2col_arrays.cc",
        "graph_transformations/ensure_bias_vectors.cc",
        "graph_transformations/fuse_activation_functions.cc",
        "graph_transformations/fuse_binary_into_following_affine.cc",
        "graph_transformations/fuse_binary_into_preceding_affine.cc",
        "graph_transformations/graph_transformations.cc",
        "graph_transformations/hardcode_min_max.cc",
        "graph_transformations/identify_l2_normalization.cc",
        "graph_transformations/identify_l2_pool.cc",
        "graph_transformations/identify_lstm.cc",
        "graph_transformations/identify_relu1.cc",
        "graph_transformations/make_initial_dequantize_operator.cc",
        "graph_transformations/propagate_array_data_types.cc",
        "graph_transformations/propagate_fixed_sizes.cc",
        "graph_transformations/quantize.cc",
        "graph_transformations/read_fake_quant_min_max.cc",
        "graph_transformations/remove_final_dequantize_op.cc",
        "graph_transformations/remove_tensorflow_assert.cc",
        "graph_transformations/remove_tensorflow_identity.cc",
        "graph_transformations/remove_trivial_binary.cc",
        "graph_transformations/remove_trivial_concatenation.cc",
        "graph_transformations/remove_trivial_concatenation_input.cc",
        "graph_transformations/remove_trivial_passthrough.cc",
        "graph_transformations/remove_trivial_passthrough.h",
        "graph_transformations/remove_trivial_quantized_activation_func.cc",
        "graph_transformations/remove_trivial_reshape.cc",
        "graph_transformations/remove_unused_op.cc",
        "graph_transformations/resolve_batch_normalization.cc",
        "graph_transformations/resolve_batch_to_space_nd_attributes.cc",
        "graph_transformations/resolve_constant_binary.cc",
        "graph_transformations/resolve_constant_concatenation.cc",
        "graph_transformations/resolve_constant_fake_quant.cc",
        "graph_transformations/resolve_constant_fill.cc",
        "graph_transformations/resolve_constant_range.cc",
        "graph_transformations/resolve_constant_shape_or_rank.cc",
        "graph_transformations/resolve_constant_stack.cc",
        "graph_transformations/resolve_constant_strided_slice.cc",
        "graph_transformations/resolve_constant_unary.cc",
        "graph_transformations/resolve_mean_attributes.cc",
        "graph_transformations/resolve_pad_attributes.cc",
        "graph_transformations/resolve_reorder_axes.cc",
        "graph_transformations/resolve_reshape_attributes.cc",
        "graph_transformations/resolve_slice_attributes.cc",
        "graph_transformations/resolve_space_to_batch_nd_attributes.cc",
        "graph_transformations/resolve_strided_slice_attributes.cc",
        "graph_transformations/resolve_tensorflow_concat.cc",
        "graph_transformations/resolve_tensorflow_matmul.cc",
        "graph_transformations/resolve_tensorflow_merge.cc",
        "graph_transformations/resolve_tensorflow_squeeze.cc",
        "graph_transformations/resolve_tensorflow_switch.cc",
        "graph_transformations/resolve_tensorflow_tile.cc",
        "graph_transformations/resolve_transpose_attributes.cc",
        "graph_transformations/unfuse_activation_functions.cc",
    ],
    hdrs = [
        "graph_transformations/graph_transformations.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
        ":model_flags_proto_cc",
        ":runtime",
        ":toco_port",
        ":tooling_util",
        ":types_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

# :toco_tooling is the library providing the offline tooling functionality
# exposed by the :toco command-line tool.
cc_library(
    name = "toco_tooling",
    srcs = [
        "allocate_transient_arrays.cc",
        "export_tensorflow.cc",
        "import_tensorflow.cc",
        "tensorflow_util.cc",
        "toco_tooling.cc",
    ],
    hdrs = [
        "allocate_transient_arrays.h",
        "export_tensorflow.h",
        "import_tensorflow.h",
        "tensorflow_util.h",
        "toco_tooling.h",
    ],
    copts = select({
        "//tensorflow:darwin": ["-DTOCO_SUPPORT_PORTABLE_PROTOS=0"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":graph_transformations",
        ":model",
        ":model_flags_proto_cc",
        ":types_proto_cc",
        ":runtime",
        ":toco_graphviz_dump_options",
        ":toco_flags_proto_cc",
        ":toco_port",
        ":tooling_util",
        "@protobuf_archive//:protobuf_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/contrib/lite/toco/tensorflow_graph_matching:resolve_cluster",
        "//tensorflow/contrib/lite/toco/tflite:export",
        "//tensorflow/contrib/lite/toco/tflite:import",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + select({
        # Placeholder for internal darwin rule.
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tooling_util",
    srcs = [
        "dump_graphviz.cc",
        "tooling_util.cc",
    ],
    hdrs = [
        "dump_graphviz.h",
        "tooling_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
        ":model_flags_proto_cc",
        ":runtime",
        ":toco_flags_proto_cc",
        ":toco_graphviz_dump_options",
        ":toco_port",
        ":types_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@protobuf_archive//:protobuf_headers",
    ],
)

tf_cc_test(
    name = "tooling_util_test",
    srcs = ["tooling_util_test.cc"],
    deps = [
        ":model",
        ":tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# :toco is the main public command-line tool exposing the functionality
# of the :toco_tooling library.
tf_cc_binary(
    name = "toco",
    srcs = ["toco.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
        ":model_cmdline_flags",
        ":model_flags_proto_cc",
        ":toco_cmdline_flags",
        ":toco_flags_proto_cc",
        ":toco_port",
        ":toco_tooling",
        ":types_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "toco_port_test",
    srcs = ["toco_port_test.cc"],
    data = [
        "toco_port_test.cc",
    ],
    deps = [
        ":toco_port",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
