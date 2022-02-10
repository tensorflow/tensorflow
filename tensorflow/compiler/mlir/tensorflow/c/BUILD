load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = ["//tensorflow/..."],
)

tf_cuda_library(
    name = "mlir_c_api",
    srcs = [
        "c_api_unified_experimental_mlir.cc",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_c_api_registration",
    srcs = ["c_api_unified_experimental_mlir_registration.cc"],
    deps = [
        ":mlir_c_api",
        "//tensorflow/c/eager:c_api_unified_internal",
    ],
    alwayslink = 1,
)
