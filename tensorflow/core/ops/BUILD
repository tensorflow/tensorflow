# Description:
# Tensorflow default op definitions.

load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "filegroup",
    "tf_gen_op_libs",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

# A lot of packages try to minimize binary size by depending on individual ops,\
# so they need access here.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

# Export the BUILD file so automated tooling can check licenses
exports_files([
    "BUILD",
    "ops.pbtxt",
])

# Generates library per group of ops.
tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "batch_ops",
        "bitwise_ops",
        "boosted_trees_ops",
        "candidate_sampling_ops",
        "checkpoint_ops",
        "clustering_ops",
        "collective_ops",
        "control_flow_ops",
        "count_ops",
        "ctc_ops",
        "data_flow_ops",
        "dataset_ops",
        "decode_proto_ops",
        "encode_proto_ops",
        "experimental_dataset_ops",
        "filesystem_ops",
        "function_ops",
        "functional_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "list_ops",
        "map_ops",
        "lookup_ops",
        "manip_ops",
        "math_ops",
        "mkl_nn_ops",
        "nccl_ops",
        "nn_ops",
        "no_op",
        "optional_ops",
        "parsing_ops",
        "random_grad",
        "random_index_shuffle_ops",
        "random_ops",
        "special_math_ops",
        "stateful_random_ops",
        "risc_ops",
        "rnn_ops",
        "scoped_allocator_ops",
        "sdca_ops",
        "set_ops",
        "script_ops",
        "sendrecv_ops",
        "sparse_ops",
        "spectral_ops",
        "state_ops",
        "stateless_random_ops",
        "stateless_random_ops_v2",
        "stochastic_cast_op",
        "sync_ops",
        "summary_ops",
        "training_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_gen_op_libs(
    is_external = False,
    op_lib_names = ["sparse_csr_matrix_ops"],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "logging_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        # TODO(b/162630222): remove this dependency.
        "//tensorflow/c/kernels:histogram_summary_op_lib",
        "//tensorflow/c/kernels:merge_summary_op_lib",
        "//tensorflow/c/kernels:summary_op_lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "string_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/strings",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "array_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "mkl_array_ops",
    ],
    sub_directory = "",
    deps = ["//tensorflow/core:protos_all_cc"],
)

tf_gen_op_libs(
    op_lib_names = [
        "audio_ops",
    ],
    sub_directory = "",
    deps = ["//tensorflow/core:lib"],
)

tf_gen_op_libs(
    op_lib_names = ["debug_ops"],
    sub_directory = "",
    deps = ["//tensorflow/core:lib"],
)

tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "resource_variable_ops",
    ],
    sub_directory = "",
    deps = ["//tensorflow/core:lib"],
)

tf_gen_op_libs(
    op_lib_names = [
        "tpu_configuration_ops",
        "tpu_cross_replica_ops",
        "tpu_embedding_ops",
        "tpu_embedding_load_retrieve_ops",
        "tpu_functional_ops",
        "tpu_heartbeat_ops",
        "tpu_host_compute_ops",
        "tpu_sharding_util_ops",
        "tpu_infeed_ops",
        "tpu_outfeed_ops",
        "tpu_ordinal_selector_ops",
        "tpu_replication_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu:tpu_embedding_output_layout_utils",
    ],
)

cc_library(
    name = "word2vec_ops",
    srcs = ["word2vec_ops.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = [
        "cudnn_rnn_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["composite_tensor_ops"],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "ragged_ops",
    deps = [
        "//tensorflow/core:ragged_array_ops_op_lib",
        "//tensorflow/core:ragged_conversion_ops_op_lib",
        "//tensorflow/core:ragged_math_ops_op_lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "ragged_array_ops",
        "ragged_conversion_ops",
        "ragged_math_ops",
    ],
    sub_directory = "",
    deps = ["//tensorflow/core/util:ragged_to_dense_util"],
)

tf_gen_op_libs(
    is_external = False,
    op_lib_names = [
        "uniform_quant_ops",
    ],
    sub_directory = "",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/quantization:uniform_quant_ops_params",
    ],
)

cc_library(
    name = "ops",
    deps = [
        ":array_ops_op_lib",
        ":audio_ops_op_lib",
        ":batch_ops_op_lib",
        ":bitwise_ops_op_lib",
        ":boosted_trees_ops_op_lib",
        ":candidate_sampling_ops_op_lib",
        ":checkpoint_ops_op_lib",
        ":clustering_ops_op_lib",
        ":collective_ops_op_lib",
        ":composite_tensor_ops_op_lib",
        ":control_flow_ops_op_lib",
        ":count_ops_op_lib",
        ":ctc_ops_op_lib",
        ":cudnn_rnn_ops_op_lib",
        ":data_flow_ops_op_lib",
        ":dataset_ops_op_lib",
        ":debug_ops_op_lib",
        ":decode_proto_ops_op_lib",
        ":encode_proto_ops_op_lib",
        ":experimental_dataset_ops_op_lib",
        ":filesystem_ops_op_lib",
        ":function_ops_op_lib",
        ":functional_ops_op_lib",
        ":image_ops_op_lib",
        ":io_ops_op_lib",
        ":linalg_ops_op_lib",
        ":list_ops_op_lib",
        ":logging_ops_op_lib",
        ":lookup_ops_op_lib",
        ":manip_ops_op_lib",
        ":map_ops_op_lib",
        ":math_ops_op_lib",
        ":nccl_ops_op_lib",
        ":nn_ops_op_lib",
        ":no_op_op_lib",
        ":optional_ops_op_lib",
        ":parsing_ops_op_lib",
        ":ragged_ops",
        ":random_index_shuffle_ops_op_lib",
        ":random_ops_op_lib",
        ":resource_variable_ops_op_lib",
        ":risc_ops_op_lib",
        ":rnn_ops_op_lib",
        ":scoped_allocator_ops_op_lib",
        ":script_ops_op_lib",
        ":sdca_ops_op_lib",
        ":sendrecv_ops_op_lib",
        ":set_ops_op_lib",
        ":sparse_csr_matrix_ops_op_lib",
        ":sparse_ops_op_lib",
        ":special_math_ops_op_lib",
        ":spectral_ops_op_lib",
        ":state_ops_op_lib",
        ":stateful_random_ops_op_lib",
        ":stateless_random_ops_op_lib",
        ":stateless_random_ops_v2_op_lib",
        ":stochastic_cast_op_op_lib",
        ":string_ops_op_lib",
        ":summary_ops_op_lib",
        ":sync_ops_op_lib",
        ":training_ops_op_lib",
        ":uniform_quant_ops_op_lib",
        ":word2vec_ops",
    ] + select({
        # Non-tpu platforms don't need tpu dependency.
        "//tensorflow:chromiumos": [],
        "//tensorflow:fuchsia": [],
        "//conditions:default": [
            ":tpu_configuration_ops_op_lib",
            ":tpu_cross_replica_ops_op_lib",
            ":tpu_embedding_load_retrieve_ops_op_lib",
            ":tpu_embedding_ops_op_lib",
            ":tpu_functional_ops_op_lib",
            ":tpu_heartbeat_ops_op_lib",
            ":tpu_host_compute_ops_op_lib",
            ":tpu_infeed_ops_op_lib",
            ":tpu_ordinal_selector_ops_op_lib",
            ":tpu_outfeed_ops_op_lib",
            ":tpu_replication_ops_op_lib",
            ":tpu_sharding_util_ops_op_lib",
        ],
    }) + if_mkl([
        ":mkl_array_ops_op_lib",
        ":mkl_nn_ops_op_lib",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "array_grad",
    srcs = ["array_grad.cc"],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    deps = [
        ":array_ops_op_lib",
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "functional_grad",
    srcs = ["functional_grad.cc"],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    deps = [
        ":functional_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "math_grad",
    srcs = [
        "math_grad.cc",
        "stateless_random_grad.cc",
    ],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    deps = [
        ":math_ops_op_lib",
        ":random_grad_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nn_grad",
    srcs = ["nn_grad.cc"],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    deps = [
        ":nn_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ] + if_mkl([
        ":mkl_nn_ops_op_lib",
    ]),
    alwayslink = 1,
)

filegroup(
    name = "portable_op_registrations_and_gradients",
    srcs = ["//tensorflow/c/kernels:portable_all_ops"] + glob(
        [
            "**/*.cc",
            "**/*.h",
        ],
        exclude = [
            "**/*test.cc",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "**/tpu_*",
        ],
    ),
)

tf_cc_test(
    name = "cudnn_rnn_ops_test_cc",
    size = "small",
    srcs = [
        "cudnn_rnn_ops_test.cc",
    ],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "ops_array_grad_test",
    size = "small",
    srcs = ["array_grad_test.cc"],
    deps = [
        ":ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "ops_math_grad_test",
    size = "small",
    srcs = ["math_grad_test.cc"],
    tags = ["no_gpu"],
    deps = [
        ":ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "ops_tests",
    size = "small",
    srcs = [
        "array_ops_test.cc",
        "candidate_sampling_ops_test.cc",
        "control_flow_ops_test.cc",
        "ctc_ops_test.cc",
        "data_flow_ops_test.cc",
        "functional_ops_test.cc",
        "image_ops_test.cc",
        "io_ops_test.cc",
        "linalg_ops_test.cc",
        "lookup_ops_test.cc",
        "math_ops_test.cc",
        "nn_ops_test.cc",
        "parsing_ops_test.cc",
        "random_ops_test.cc",
        "rnn_ops_test.cc",
        "set_ops_test.cc",
        "shape_function_test.cc",
        "sparse_csr_matrix_ops_test.cc",
        "sparse_ops_test.cc",
        "spectral_ops_test.cc",
        "state_ops_test.cc",
        "stochastic_cast_op_test.cc",
        "string_ops_test.cc",
        "tpu_cross_replica_ops_test.cc",
        "training_ops_test.cc",
        "uniform_quant_ops_test.cc",
    ],
    deps = [
        ":ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:type_inference",
    ],
)
