load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/serving/contrib/tfrt/mlir/saved_model_analysis:__pkg__",
        "//tensorflow/compiler/mlir:__pkg__",
        "//tensorflow/compiler/mlir/tensorflow/transforms:__pkg__",
        "//tensorflow/compiler/mlir/tf2xla/api:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/compiler/tf2xla:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "lower_cluster_to_runtime_ops",
    srcs = ["lower_cluster_to_runtime_ops.cc"],
    hdrs = [
        "lower_cluster_to_runtime_ops.h",
    ],
    deps = [
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/tensorflow:bridge_logger",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:error_logging",
        "@local_tsl//tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "lower_cluster_to_runtime_ops_test",
    srcs = ["lower_cluster_to_runtime_ops_test.cc"],
    data = [
        "testdata/basic_cluster.mlir",
        "testdata/empty_func.mlir",
        "testdata/malformed_cluster.mlir",
    ],
    deps = [
        ":lower_cluster_to_runtime_ops",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
)
