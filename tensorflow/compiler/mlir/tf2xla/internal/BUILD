load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tensorflow/transforms:__pkg__",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:__subpackages__",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:__subpackages__",
    ],
)

cc_library(
    name = "compilation_timer",
    hdrs = ["compilation_timer.h"],
    deps = [
        "//tensorflow/core/platform:profile_utils_cpu_utils",
    ],
)

tf_cc_test(
    name = "compilation_timer_test",
    srcs = ["compilation_timer_test.cc"],
    deps = [
        ":compilation_timer",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    hdrs = ["test_matchers.h"],
    deps = [
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "test_matchers_test",
    srcs = ["test_matchers_test.cc"],
    deps = [
        ":test_matchers",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "mlir_pass_instrumentation",
    srcs = ["mlir_pass_instrumentation.cc"],
    hdrs = ["mlir_pass_instrumentation.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/log",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "mlir_pass_instrumentation_test",
    srcs = ["mlir_pass_instrumentation_test.cc"],
    deps = [
        ":mlir_pass_instrumentation",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "legalize_tf_mlir",
    srcs = ["legalize_tf_mlir.cc"],
    hdrs = ["legalize_tf_mlir.h"],
    deps = [
        ":compilation_timer",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow/transforms:set_tpu_infeed_layout",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_compile",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:error_logging",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "legalize_tf_mlir_test",
    srcs = ["legalize_tf_mlir_test.cc"],
    deps = [
        ":legalize_tf_mlir",
        ":test_matchers",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:errors",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "legalize_tf_to_hlo",
    srcs = ["legalize_tf_to_hlo.cc"],
    hdrs = ["legalize_tf_to_hlo.h"],
    deps = [
        ":legalize_tf_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow/transforms:set_tpu_infeed_layout",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_tf_graph",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_tpu_backend_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "legalize_tf_to_hlo_test",
    srcs = ["legalize_tf_to_hlo_test.cc"],
    deps = [
        ":legalize_tf_to_hlo",
        ":test_matchers",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/tsl/platform:errors",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "clustering_bridge_passes",
    srcs = ["clustering_bridge_passes.cc"],
    hdrs = ["clustering_bridge_passes.h"],
    deps = [
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//tensorflow/compiler/mlir/tensorflow/transforms/sparsecore:sparsecore_passes",
        "//tensorflow/compiler/mlir/tf2xla/internal/passes:clustering_passes",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "clustering_bridge_passes_test",
    srcs = ["clustering_bridge_passes_test.cc"],
    deps = [
        ":clustering_bridge_passes",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "logging_hooks",
    srcs = ["logging_hooks.cc"],
    hdrs = ["logging_hooks.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:bridge_logger",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "logging_hooks_test",
    srcs = ["logging_hooks_test.cc"],
    data = [
        "testdata/dead_const.mlir",
    ],
    deps = [
        ":logging_hooks",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "mlir_bridge_pass_util",
    srcs = ["mlir_bridge_pass_util.cc"],
    hdrs = ["mlir_bridge_pass_util.h"],
    visibility = ["//tensorflow/compiler/tf2xla:__pkg__"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/tf2xla:tf2xla_defs",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:function_body",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "mlir_bridge_pass_util_test",
    srcs = ["mlir_bridge_pass_util_test.cc"],
    deps = [
        ":mlir_bridge_pass_util",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:tpu_ops",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/tf2xla:tf2xla_defs",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:enable_tf2_utils",
        # "//tensorflow/core/platform:resource_loader",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_proto_library(
    name = "reproducer_proto",
    srcs = ["reproducer.proto"],
    protodeps = [
        "//tensorflow/core:protos_all",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto",
    ],
    visibility = [
        "//learning/brain/mlir/bridge:__pkg__",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:__pkg__",
    ],
)

cc_library(
    name = "node_order",
    srcs = ["node_order.cc"],
    hdrs = ["node_order.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "node_order_test",
    size = "small",
    srcs = [
        "node_order_test.cc",
    ],
    deps = [
        ":node_order",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "graph_to_tf_executor_util",
    srcs = ["graph_to_tf_executor_util.cc"],
    hdrs = ["graph_to_tf_executor_util.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:function_body",
        "//tensorflow/core/platform:enable_tf2_utils",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_cc_test(
    name = "graph_to_tf_executor_util_test",
    srcs = ["graph_to_tf_executor_util_test.cc"],
    deps = [
        ":graph_to_tf_executor_util",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:tpu_ops",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:enable_tf2_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
    ],
)
