load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_to_tf_executor_registration",
    srcs = [
        "graph_to_tf_executor_registration.cc",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/tools:translate_cl_options",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/translate:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/translate:translate_lib",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/service/cpu:cpu_compiler",
        "@local_xla//xla/service/cpu:cpu_transfer_manager",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "graph_to_tf_executor_registration_test",
    size = "small",
    srcs = ["graph_to_tf_executor_registration_test.cc"],
    deps = [
        ":graph_to_tf_executor_registration",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
