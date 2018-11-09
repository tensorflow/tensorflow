# Description: Utilities for TPU Operations

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "tpu_embedding_optimization_parameters_utils",
    srcs = ["tpu_embedding_optimization_parameters_utils.cc"],
    hdrs = ["tpu_embedding_optimization_parameters_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/tpu/proto:optimization_parameters_proto_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "tpu_embedding_output_layout_utils",
    srcs = ["tpu_embedding_output_layout_utils.cc"],
    hdrs = ["tpu_embedding_output_layout_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/tpu/proto:tpu_embedding_configuration_proto_cc",
        "//tensorflow/contrib/tpu/proto:tpu_embedding_output_layout_proto_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
    ],
)
