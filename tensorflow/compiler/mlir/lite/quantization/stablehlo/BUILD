load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/mlir/lite:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "quantization",
    srcs = ["quantization.cc"],
    hdrs = ["quantization.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/lite/stablehlo:tf_stablehlo",
        "//tensorflow/compiler/mlir/quantization/stablehlo:passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:config",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:static_range_ptq",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:weight_only_ptq",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_freeze_variables",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "quantization_test",
    srcs = ["quantization_test.cc"],
    deps = [
        ":quantization",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:io",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton_impl",  # buildcleaner: keep; prevents undefined reference
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)
