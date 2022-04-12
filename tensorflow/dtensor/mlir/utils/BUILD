# DTensor Internal experimental code.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        "//tensorflow/dtensor:dtensor-internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dtensor_mlir_passes_internal",
    srcs = [
        "collective_lowering.cc",
        "dtensor_embedding_stub.cc",
        "dtensor_embedding_v2_stub.cc",
        "dtensor_mlir_passes_internal.cc",
        "update_tpu_metadata.cc",
    ],
    hdrs = ["dtensor_mlir_passes_internal.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/dtensor/cc:constants",
        "//tensorflow/dtensor/cc:dstatus",
        "//tensorflow/dtensor/cc:dtensor_utils",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/mlir:collectives_common",
        "//tensorflow/dtensor/mlir:create_dtensor_mlir_passes",
        "//tensorflow/dtensor/mlir:device_utils",
        "//tensorflow/dtensor/mlir:dtensor_location",
        "//tensorflow/dtensor/mlir:group_assignment",
        "//tensorflow/dtensor/mlir:layout_parsing",
        "//tensorflow/dtensor/mlir:op_utils",
        "//tensorflow/dtensor/mlir:spmd_expander_common",
        "//tensorflow/dtensor/mlir:tf_dtensor_dialect",
        "//tensorflow/dtensor/mlir:value_utils",
        "//tensorflow/dtensor/mlir/dtensor_dialect:ir/dtensor_attributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
