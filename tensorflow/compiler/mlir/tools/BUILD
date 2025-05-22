package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "translate_cl_options",
    srcs = [
        "tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "tf_mlir_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translate_registration",
    srcs = [
        "tf_mlir_translate_registration.cc",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/translate:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/translate/tools:file_tf_mlir_translate",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/compiler/mlir/tools:translate_cl_options",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/service/cpu:cpu_compiler",
        "@local_xla//xla/service/cpu:cpu_transfer_manager",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)
