load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":__subpackages__",
        ":tf2xla_users",
    ],
)

# Please reach out to tf-bridge-team@ before using the TF2XLA bridge.
package_group(
    name = "tf2xla_users",
    packages = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/...",
        "//learning/serving/contrib/tfrt/mlir/saved_model_analysis",
        "//tensorflow/compiler/mlir/tfrt",
        "//tensorflow/compiler/tf2xla",
        # Legacy due to where the bridge currently runs. This should go away.
        "//tensorflow/compiler/mlir/tensorflow/transforms",
    ],
)

cc_library(
    name = "legalize_tf",
    srcs = ["legalize_tf.cc"],
    hdrs = ["legalize_tf.h"],
    deps = [
        ":device_type_proto_cc",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/mlir/tensorflow/transforms:set_tpu_infeed_layout",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_tf_graph",
        "//tensorflow/compiler/mlir/tf2xla/internal:legalize_tf_mlir",
        "//tensorflow/compiler/mlir/tf2xla/internal:legalize_tf_to_hlo",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/tpu:tpu_compile",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "//tensorflow/core/tpu/kernels:tpu_compile_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_util_hdrs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:error_logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/pjrt:compile_options_proto_cc",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "legalize_tf_test",
    srcs = ["legalize_tf_test.cc"],
    deps = [
        ":device_type_proto_cc",
        ":legalize_tf",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/monitoring:test_utils",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/client:client_library",
    ],
)

tf_proto_library(
    name = "device_type_proto",
    srcs = ["device_type.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "cluster_tf",
    srcs = ["cluster_tf.cc"],
    hdrs = ["cluster_tf.h"],
    deps = [
        ":device_type_proto_cc",
        ":tf_dialect_to_executor",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//tensorflow/compiler/mlir/tf2xla/internal:clustering_bridge_passes",
        "//tensorflow/compiler/mlir/tf2xla/internal:logging_hooks",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:stacktrace",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:error_logging",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

tf_cc_test(
    name = "cluster_tf_test",
    srcs = ["cluster_tf_test.cc"],
    data = [
        "testdata/empty_func.mlir",
        "testdata/invalid_executor.mlir",
        "testdata/outside_compilation.mlir",
    ],
    deps = [
        ":cluster_tf",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tf_dialect_to_executor",
    srcs = ["tf_dialect_to_executor.cc"],
    hdrs = ["tf_dialect_to_executor.h"],
    deps = [
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/tensorflow:bridge_logger",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//tensorflow/compiler/mlir/tf2xla/internal:logging_hooks",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/lib/monitoring:counter",
        "@local_tsl//tsl/platform:error_logging",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

tf_cc_test(
    name = "tf_dialect_to_executor_test",
    srcs = ["tf_dialect_to_executor_test.cc"],
    data = [
        "testdata/empty_func.mlir",
        "testdata/invalid_executor.mlir",
    ],
    deps = [
        ":tf_dialect_to_executor",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/lib/monitoring:test_utils",
        "@local_tsl//tsl/platform:status",
    ],
)
