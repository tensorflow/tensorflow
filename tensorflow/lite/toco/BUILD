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
    "tf_copts",
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
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:types",
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
        "@com_google_absl//absl/types:optional",
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
        "@com_google_absl//absl/types:optional",
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
        "@protobuf_archive//:protobuf_headers",
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
        "graph_transformations/convert_reorder_axes.cc",
        "graph_transformations/convert_squeeze_to_reshape.cc",
        "graph_transformations/convert_trivial_addn_to_add.cc",
        "graph_transformations/convert_trivial_pack_to_reshape.cc",
        "graph_transformations/convert_trivial_tile_to_concat.cc",
        "graph_transformations/convert_trivial_transpose_to_reshape.cc",
        "graph_transformations/create_im2col_arrays.cc",
        "graph_transformations/dequantize.cc",
        "graph_transformations/drop_fake_quant.cc",
        "graph_transformations/drop_im2col_arrays.cc",
        "graph_transformations/ensure_bias_vectors.cc",
        "graph_transformations/ensure_uint8_weights_safe_for_fast_int8_kernels.cc",
        "graph_transformations/fuse_activation_functions.cc",
        "graph_transformations/fuse_binary_into_following_affine.cc",
        "graph_transformations/fuse_binary_into_preceding_affine.cc",
        "graph_transformations/fuse_broadcast_into_following_binary.cc",
        "graph_transformations/graph_transformations.cc",
        "graph_transformations/hardcode_min_max.cc",
        "graph_transformations/identify_dilated_conv.cc",
        "graph_transformations/identify_l2_normalization.cc",
        "graph_transformations/identify_l2_pool.cc",
        "graph_transformations/identify_lstm.cc",
        "graph_transformations/identify_lstm_merge_inputs.cc",
        "graph_transformations/identify_lstm_split_inputs.cc",
        "graph_transformations/identify_prelu.cc",
        "graph_transformations/identify_relu1.cc",
        "graph_transformations/lstm_utils.cc",
        "graph_transformations/make_initial_dequantize_operator.cc",
        "graph_transformations/merge_reshape_into_preceding_transpose.cc",
        "graph_transformations/move_binary_operator_before_reshape.cc",
        "graph_transformations/propagate_activation_function_into_constants.cc",
        "graph_transformations/propagate_array_data_types.cc",
        "graph_transformations/propagate_default_min_max.cc",
        "graph_transformations/propagate_fake_quant_num_bits.cc",
        "graph_transformations/propagate_fixed_sizes.cc",
        "graph_transformations/quantization_util.cc",
        "graph_transformations/quantization_util.h",
        "graph_transformations/quantize.cc",
        "graph_transformations/read_array_minmax_and_narrow_range_from_fake_quant.cc",
        "graph_transformations/remove_final_dequantize_op.cc",
        "graph_transformations/remove_tensorflow_assert.cc",
        "graph_transformations/remove_tensorflow_identity.cc",
        "graph_transformations/remove_trivial_binary.cc",
        "graph_transformations/remove_trivial_concatenation.cc",
        "graph_transformations/remove_trivial_concatenation_input.cc",
        "graph_transformations/remove_trivial_fake_quant.cc",
        "graph_transformations/remove_trivial_passthrough.cc",
        "graph_transformations/remove_trivial_passthrough.h",
        "graph_transformations/remove_trivial_quantized_activation_func.cc",
        "graph_transformations/remove_trivial_quantized_min_max.cc",
        "graph_transformations/remove_trivial_reshape.cc",
        "graph_transformations/remove_trivial_slice.cc",
        "graph_transformations/remove_unused_op.cc",
        "graph_transformations/reorder_elementwise_unary.cc",
        "graph_transformations/reorder_reshape_transpose.cc",
        "graph_transformations/resolve_batch_normalization.cc",
        "graph_transformations/resolve_batch_to_space_nd_attributes.cc",
        "graph_transformations/resolve_constant_binary.cc",
        "graph_transformations/resolve_constant_concatenation.cc",
        "graph_transformations/resolve_constant_fake_quant.cc",
        "graph_transformations/resolve_constant_fill.cc",
        "graph_transformations/resolve_constant_gather.cc",
        "graph_transformations/resolve_constant_pack.cc",
        "graph_transformations/resolve_constant_random_uniform.cc",
        "graph_transformations/resolve_constant_range.cc",
        "graph_transformations/resolve_constant_reshape.cc",
        "graph_transformations/resolve_constant_select.cc",
        "graph_transformations/resolve_constant_shape_or_rank.cc",
        "graph_transformations/resolve_constant_slice.cc",
        "graph_transformations/resolve_constant_strided_slice.cc",
        "graph_transformations/resolve_constant_tile.cc",
        "graph_transformations/resolve_constant_transpose.cc",
        "graph_transformations/resolve_constant_unary.cc",
        "graph_transformations/resolve_fake_quant_args_from_vars.cc",
        "graph_transformations/resolve_gather_attributes.cc",
        "graph_transformations/resolve_multiply_by_zero.cc",
        "graph_transformations/resolve_pad_attributes.cc",
        "graph_transformations/resolve_padv2_attributes.cc",
        "graph_transformations/resolve_reduce_attributes.cc",
        "graph_transformations/resolve_reorder_axes.cc",
        "graph_transformations/resolve_reshape_attributes.cc",
        "graph_transformations/resolve_slice_attributes.cc",
        "graph_transformations/resolve_space_to_batch_nd_attributes.cc",
        "graph_transformations/resolve_squeeze_attributes.cc",
        "graph_transformations/resolve_strided_slice_attributes.cc",
        "graph_transformations/resolve_tensorflow_concat.cc",
        "graph_transformations/resolve_tensorflow_matmul.cc",
        "graph_transformations/resolve_tensorflow_merge.cc",
        "graph_transformations/resolve_tensorflow_switch.cc",
        "graph_transformations/resolve_transpose_attributes.cc",
        "graph_transformations/shuffle_fc_weights.cc",
        "graph_transformations/unfuse_activation_functions.cc",
        "graph_transformations/unpartition_embedding_lookup.cc",
        "graph_transformations/unroll_batch_matmul.cc",
    ],
    hdrs = [
        "graph_transformations/graph_transformations.h",
        "graph_transformations/lstm_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
        ":model_flags_proto_cc",
        ":runtime",
        ":toco_port",
        ":tooling_util",
        "//tensorflow/core:lib",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:strided_slice_logic",
        "@com_google_absl//absl/memory",
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
    copts = tf_copts() + select({
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
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco/tensorflow_graph_matching:resolve_cluster",
        "//tensorflow/lite/toco/tflite:export",
        "//tensorflow/lite/toco/tflite:import",
    ] + select({
        # Placeholder for internal darwin rule.
        "//conditions:default": [],
    }),
)

tf_cc_test(
    name = "import_tensorflow_test",
    srcs = ["import_tensorflow_test.cc"],
    deps = [
        ":toco_tooling",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
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
    copts = tf_copts(),
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
        "//tensorflow/lite/kernels/internal:types",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@protobuf_archive//:protobuf_headers",
    ],
)

tf_cc_test(
    name = "tooling_util_test",
    srcs = ["tooling_util_test.cc"],
    deps = [
        ":model",
        ":tooling_util",
        "//tensorflow/core:lib",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# :toco is the main public command-line tool exposing the functionality
# of the :toco_tooling library.
cc_library(
    name = "toco_convert",
    srcs = ["toco_convert.cc"],
    hdrs = ["toco_convert.h"],
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
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        # We cannot embed the core:ops dependency directly into :toco_tooling as
        # it can conflict with downstream deps when toco is used as a library.
        "//tensorflow/core:ops",
    ],
)

tf_cc_binary(
    name = "toco",
    srcs = ["toco.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
        ":model_cmdline_flags",
        ":model_flags_proto_cc",
        ":toco_cmdline_flags",
        ":toco_convert",
        ":toco_flags_proto_cc",
        ":toco_port",
        ":toco_tooling",
        ":types_proto_cc",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        # We cannot embed the core:ops dependency directly into :toco_tooling as
        # it can conflict with downstream deps when toco is used as a library.
        "//tensorflow/core:ops",
    ],
)

tf_cc_test(
    name = "toco_convert_test",
    srcs = ["toco_convert_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
        ":model_cmdline_flags",
        ":model_flags_proto_cc",
        ":toco_cmdline_flags",
        ":toco_convert",
        ":toco_flags_proto_cc",
        ":toco_port",
        ":toco_tooling",
        ":types_proto_cc",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        # We cannot embed the core:ops dependency directly into :toco_tooling as
        # it can conflict with downstream deps when toco is used as a library.
        "//tensorflow/core:ops",
        "//tensorflow/lite/testing:util",
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
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
