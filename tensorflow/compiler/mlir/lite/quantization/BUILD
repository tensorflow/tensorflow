load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["@local_config_mlir//:subpackages"],
    packages = ["//tensorflow/compiler/mlir/..."],
)

exports_files(["quantization_traits.h"])

filegroup(
    name = "quantization_td_files",
    srcs = [
        "quantization.td",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:QuantizationOpsTdFiles",
    ],
)

cc_library(
    name = "quantization_lib",
    srcs = [
        "quantization_driver.cc",
        "quantization_utils.cc",
    ],
    hdrs = [
        "quantization_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        # TODO(fengliuai): remove this dependence.
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/core:lib_proto_parsing",
    ],
)

tf_native_cc_binary(
    name = "op_quant_spec_getters_gen",
    srcs = [
        "tools/op_quant_spec_getters_gen.cc",
    ],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:TableGen",
    ],
)
