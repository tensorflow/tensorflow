# Description:
#   TensorFlow/TensorFlow Lite/XLA MLIR dialects and tools.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:__subpackages__",
        "@local_config_mlir//:friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "subpackages",
    packages = ["//tensorflow/compiler/mlir/..."],
)

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "td_files",
    srcs = glob(["**/*.td"]),
)

cc_library(
    name = "op_name_mapper",
    srcs = ["op_name_mapper.cc"],
    hdrs = ["op_name_mapper.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "tf_mlir_opt_main",
    srcs = ["tf_mlir_opt_main.cc"],
    deps = [
        ":init_mlir",
        ":passes",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm//:support",
        "@local_config_mlir//:MlirOptLib",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
        "@local_config_mlir//test:TestTransforms",
    ],
)

cc_library(
    name = "passes",
    visibility = [
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_dialect_registration",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_optimize",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_test_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/mlir/xla:hlo_legalize_to_lhlo",
        "//tensorflow/compiler/mlir/xla:lhlo",
        "//tensorflow/compiler/mlir/xla:lhlo_fuse_linalg",
        "//tensorflow/compiler/mlir/xla:lhlo_legalize_to_affine",
        "//tensorflow/compiler/mlir/xla:lhlo_legalize_to_linalg",
        "//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "//tensorflow/compiler/mlir/xla:xla_legalize_control_flow",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_legalize_to_standard",
        "//tensorflow/compiler/mlir/xla:xla_lower_general_dot",
        "@local_config_mlir//:AffineDialectRegistration",
        "@local_config_mlir//:QuantOps",
        "@local_config_mlir//:QuantOpsDialectRegistration",
    ],
)

cc_library(
    name = "init_mlir",
    srcs = ["init_mlir.cc"],
    hdrs = ["init_mlir.h"],
    deps = [
        "//tensorflow/core:lib",
        "@llvm//:support",
    ],
)

tf_cc_binary(
    name = "tf-opt",
    deps = [
        ":tf_mlir_opt_main",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
    ],
)

tf_cc_binary(
    name = "tf-mlir-translate",
    srcs = ["tf_mlir_translate_main.cc"],
    deps = [
        ":init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow:translate_registration",
        "//tensorflow/compiler/mlir/tensorflow:translate_tf_dialect_op",
        "//tensorflow/compiler/mlir/xla:xla_mlir_translate",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TranslateClParser",
        "@local_config_mlir//:Translation",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
)
