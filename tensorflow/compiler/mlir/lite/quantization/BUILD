load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library",
)

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

tf_proto_library(
    name = "quantization_info_proto",
    srcs = [
        "quantization_info.proto",
    ],
    cc_api_version = 2,
)

cc_library(
    name = "quantization_passes",
    srcs = [
        "import_quant_stats_pass.cc",
    ],
    hdrs = [
        "quantization_passes.h",
    ],
    deps = [
        ":quantization_info_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
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

cc_library(
    name = "quantization_config",
    srcs = [
        "quantization_config.cc",
    ],
    hdrs = [
        "quantization_config.h",
    ],
    deps = [
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
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
