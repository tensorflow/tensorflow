# TPU Kernel Implementations

load("@rules_python//python:proto.bzl", "py_proto_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
    "if_oss",
    "tf_cc_test",
    "tf_copts",
    "tf_gen_op_wrapper_py",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "tf_grpc_cc_dependencies",
    "tf_kernel_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/tpu:build_defs.bzl",
    "if_libtpu_tf_status",
    "if_libtpu_tf_tensor",
)

# Config setting to enable go/libtpu support.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [] + if_google([
        "//learning/brain/google/xla:friends",
    ]),
    packages = [
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tf2xla/...",
        "//tensorflow/core/tfrt/ifrt/...",
        "//tensorflow/core/tpu/...",
        "//tensorflow/dtensor/...",
        "//third_party/py/jax_tpu_embedding/...",
    ] + if_google([
        "//learning/brain/google/xla/...",
    ]),
)

tf_kernel_library(
    name = "kernels",
    visibility = ["//visibility:public"],
    deps = [
        ":cross_replica_ops",
        ":global_iter_id_op",
        ":host_compute_ops",
        ":image_resize_ops",
        ":infeed_ops",
        ":outfeed_ops",
        ":replication_ops",
        ":sharding_util_ops",
        ":sparse_core_preprocess_ops",
        ":sparse_core_xla_ops",
        ":topk_ops",
        ":tpu_configuration_ops",
        ":tpu_dummy_input_op",
        ":tpu_embedding_configuration_ops",
        ":tpu_embedding_enqueue_ops",
        ":tpu_embedding_load_retrieve_ops",
        ":tpu_embedding_ops",
        ":tpu_functional_ops",
        ":tpu_ordinal_selector_op",
        ":transfer_ops",
    ],
)

cc_library(
    name = "tpu_compile_op_options",
    srcs = ["tpu_compile_op_options.cc"],
    hdrs = ["tpu_compile_op_options.h"],
)

cc_library(
    name = "sparse_core_preprocess_ops",
    srcs = ["sparse_core_preprocess_ops.cc"],
    hdrs = ["sparse_core_preprocess_ops.h"],
    deps = [
        ":sparse_core_ops_stats_handler",
        ":sparse_core_ops_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_highway//:hwy",
        "@com_google_highway//hwy/contrib/sort:vqsort",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:util",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "sparse_core_xla_ops",
    srcs = ["sparse_core_xla_ops.cc"],
    hdrs = ["sparse_core_xla_ops.h"],
    deps = [
        ":sparse_core_ops_utils",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/builder/lib:arithmetic",
        "@xla//xla/hlo/builder/lib:slicing",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
        "@xla//xla/tsl/platform:errors",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sparse_core_ops_stats_handler",
    hdrs = ["sparse_core_ops_stats_handler.h"],
)

tf_kernel_library(
    name = "tpu_configuration_ops",
    srcs = ["tpu_configuration_ops.cc"],
    hdrs = ["tpu_configuration_ops.h"],
    deps =
        [
            ":tpu_embedding_engine_state_interface",
            ":tpu_execute_op_options",
            ":tpu_fingerprint_lookup",
            ":tpu_mesh_state_interface",
            ":tpu_op_consts",
            ":tpu_util",
            "//tensorflow/c:tf_status_helper",
            "//tensorflow/core:core_cpu_base",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:refcount",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/platform:statusor",
            "//tensorflow/core/tpu:tpu_configuration",
            "//tensorflow/core/tpu:tpu_defs",
            "@com_google_absl//absl/cleanup",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/strings",
            "@tsl//tsl/platform:tstring",
            "@xla//xla:util",
            "@xla//xla/stream_executor:stream",
            "@xla//xla/stream_executor:stream_executor_h",
            "@xla//xla/tpu:proto_helper",
            "@xla//xla/tpu:status_helper",
            "@xla//xla/tpu:tpu_api",
            "@xla//xla/tpu:tpu_ops_c_api_hdrs",
            "@xla//xla/tsl/platform:errors",
            "@xla//xla/tsl/platform:logging",
            "@xla//xla/tsl/platform:macros",
            "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        ] + if_libtpu_tf_status(),
    alwayslink = 1,
)

tf_kernel_library(
    name = "tpu_dummy_input_op",
    srcs = ["tpu_dummy_input_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ops:tpu_replication_ops_op_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:bfloat16",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "tpu_dummy_input_op_test",
    srcs = ["tpu_dummy_input_op_test.cc"],
    deps = [
        ":tpu_dummy_input_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:bfloat16",
        "@xla//xla/tsl/platform:errors",
    ],
)

tf_kernel_library(
    name = "tpu_embedding_configuration_ops",
    srcs = ["tpu_embedding_configuration_ops.cc"],
    deps =
        [
            ":tpu_embedding_engine_state_interface",
            ":tpu_execute_op_options",
            ":tpu_fingerprint_lookup",
            ":tpu_mesh_state_interface",
            ":tpu_op_consts",
            ":tpu_util",
            "//tensorflow/c:tf_status_helper",
            "//tensorflow/core:core_cpu_base",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:refcount",
            "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
            "//tensorflow/core/tpu:tpu_configuration",
            "//tensorflow/core/tpu:tpu_defs",
            "@com_google_absl//absl/cleanup",
            "@xla//xla:util",
            "@xla//xla/tpu:proto_helper",
            "@xla//xla/tpu:status_helper",
            "@xla//xla/tpu:tpu_api",
            "@xla//xla/tpu:tpu_ops_c_api_hdrs",
        ] + if_libtpu_tf_status(),
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_load_retrieve_ops",
    srcs = ["tpu_embedding_load_retrieve_ops.cc"],
    hdrs = ["tpu_embedding_load_retrieve_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_configuration_proto_rewrite",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_shape_util",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tpu:c_api_conversions",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:status_helper",
        "@xla//xla/tpu:tpu_api",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_enqueue_ops",
    srcs = ["tpu_embedding_enqueue_ops.cc"],
    hdrs = ["tpu_embedding_enqueue_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:status_helper",
        "@xla//xla/tpu:tpu_api",
    ] + if_libtpu_tf_tensor(),
    alwayslink = 1,
)

tf_proto_library(
    name = "tpu_executable_info_proto",
    srcs = ["tpu_executable_info.proto"],
    protodeps = [
        "@xla//xla:xla_data_proto",
        "@xla//xla/service:hlo_proto",
        "//tensorflow/core:protos_all",
    ],
)

cc_library(
    name = "tpu_compile_op_support",
    srcs = ["tpu_compile_op_support.cc"],
    hdrs = ["tpu_compile_op_support.h"],
    visibility = [
        ":friends",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@xla//xla:debug_options_flags",
        "@xla//xla:shape_tree",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:computation_layout",
        "@xla//xla/service:computation_placer",
        "@xla//xla/service:dump",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tpu_embedding_engine_state_interface",
    srcs = [],
    hdrs = ["tpu_embedding_engine_state_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_mesh_state_interface",
    srcs = [],
    hdrs = ["tpu_mesh_state_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_compilation_metrics_hdrs",
    hdrs = ["tpu_compilation_metrics.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "tpu_compilation_metrics",
    srcs = ["tpu_compilation_metrics.cc"],
    copts = tf_copts(),
    deps = [
        ":tpu_compilation_metrics_hdrs",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_util_hdrs",
    hdrs = ["trace_util.h"],
)

cc_library(
    name = "tpu_util_hdrs",
    hdrs = ["tpu_util.h"],
    deps = [
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@xla//xla/client:compile_only_client",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "tpu_op_util",
    srcs = ["tpu_op_util.cc"],
    hdrs = ["tpu_op_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_util",
    srcs = ["tpu_util.cc"],
    hdrs = ["tpu_util.h"],
    deps = [
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@xla//xla:shape_util",
        "@xla//xla/client:compile_only_client",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

cc_library(
    name = "tpu_op_consts",
    srcs = ["tpu_op_consts.cc"],
    hdrs = ["tpu_op_consts.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "tpu_execute_op_options",
    srcs = ["tpu_execute_op_options.cc"],
    hdrs = ["tpu_execute_op_options.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "cross_replica_ops",
    srcs = ["cross_replica_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@xla//xla/hlo/builder:xla_builder",
    ],
    alwayslink = 1,
)

cc_library(
    name = "topk_ops",
    srcs = ["topk_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/numeric:bits",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder/lib:arithmetic",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_ops",
    srcs = ["tpu_embedding_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:if_op",
        "//tensorflow/compiler/tf2xla/kernels:while_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_spmd_sharding_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/tpu:c_api_conversions",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:proto_helper",
        "@xla//xla/tpu:status_helper",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "host_compute_ops",
    srcs = ["host_compute_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/tpu:tpu_defs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "infeed_ops",
    srcs = ["infeed_ops.cc"],
    hdrs = ["infeed_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":transfer_ops",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/kernels:transpose_functor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla/tpu:c_api_conversions",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:noncopyable_buffer",
        "@xla//xla/tpu:tpu_executor_api",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "transfer_ops",
    srcs = ["transfer_ops.cc"],
    hdrs = ["transfer_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@xla//xla:literal",
        "@xla//xla/tpu:noncopyable_buffer",
    ],
    alwayslink = True,
)

cc_library(
    name = "outfeed_ops",
    hdrs = ["outfeed_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":transfer_ops",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "@com_google_absl//absl/status",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "image_resize_ops",
    srcs = ["image_resize_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder/lib:constants",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "replication_ops",
    srcs = ["replication_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_ordinal_selector_op",
    srcs = ["tpu_ordinal_selector_op.cc"],
    deps = [
        ":tpu_ordinal_selector",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_ordinal_selector_interface",
    hdrs = ["tpu_ordinal_selector_interface.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tpu_ordinal_selector",
    hdrs = ["tpu_ordinal_selector.h"],
    deps = [
        ":tpu_ordinal_selector_interface",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_fingerprint_lookup",
    srcs = ["tpu_fingerprint_lookup.cc"],
    hdrs = ["tpu_fingerprint_lookup.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_functional_ops",
    srcs = ["tpu_functional_ops.cc"],
    hdrs = ["tpu_functional_ops.h"],
    deps = [
        ":tpu_compile_op_support",
        ":tpu_fingerprint_lookup",
        ":tpu_op_consts",
        ":tpu_op_util",
        ":tpu_ordinal_selector",
        ":tpu_util_hdrs",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/common_runtime:placer",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:array4d",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
        "@xla//xla/tpu:tpu_topology_external",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ] + if_static(["//tensorflow/core/common_runtime:rendezvous_mgr"]),
    alwayslink = 1,
)

cc_library(
    name = "sharding_util_ops",
    srcs = ["sharding_util_ops.cc"],
    deps = [
        ":sharding_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sharding_utils",
    srcs = ["sharding_utils.cc"],
    hdrs = ["sharding_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "sharding_utils_test",
    srcs = ["sharding_utils_test.cc"],
    deps = [
        ":sharding_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

tf_kernel_library(
    name = "global_iter_id_op",
    srcs = ["global_iter_id.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:partitioned_function_ops",
        "//tensorflow/core/tpu/ops:sparse_core_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_global_iter_id_op",
    out = "gen_global_iter_id_op.py",
    op_allowlist = [
        "GlobalIterId",
    ],
    visibility = ["//visibility:public"],
    deps = [":global_iter_id_op"],
)

tf_cc_test(
    name = "sharding_util_ops_test",
    srcs = ["sharding_util_ops_test.cc"],
    deps = [
        ":sharding_util_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "sparse_core_ops_utils",
    srcs = ["sparse_core_ops_utils.cc"],
    hdrs = ["sparse_core_ops_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/tpu:status_helper",
        "@xla//xla/tpu:tpu_api",
        "@xla//xla/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "sparse_core_xla_flags_defaults",
    hdrs = ["sparse_core_xla_flags_defaults.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "_pywrap_sparse_core_layout_header_only",
    srcs = [],
    hdrs = ["sparse_core_layout.h"],
    visibility = ["//tensorflow/python/tpu:__pkg__"],  # ONLY for `_pywrap_sparse_core_layout`.
    deps = [
        ":sparse_core_layout_proto_cc",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "sparse_core_layout",
    srcs = ["sparse_core_layout.cc"],
    hdrs = ["sparse_core_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sparse_core_layout_proto_cc",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:stringpiece",
    ],
)

tf_cc_test(
    name = "sparse_core_layout_test",
    srcs = ["sparse_core_layout_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":sparse_core_layout",
        ":sparse_core_layout_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "sparse_core_ops_utils_test",
    srcs = ["sparse_core_ops_utils_test.cc"],
    deps = [
        ":sparse_core_ops_utils",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:xla_data_proto_cc",
    ],
)

tf_proto_library(
    name = "sparse_core_layout_proto",
    srcs = ["sparse_core_layout.proto"],
    has_services = False,
    create_go_proto = False,
    create_java_proto = False,
    create_kotlin_proto = False,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "sparse_core_layout_py_pb2",
#     deps = [":sparse_core_layout_proto"],
# )
# copybara:uncomment_end
