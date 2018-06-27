licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_proto_library_py",
)

tf_proto_library(
    name = "tpu_embedding_config_proto",
    srcs = [
        "tpu_embedding_config.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "topology_proto",
    srcs = [
        "topology.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "compilation_result_proto",
    srcs = [
        "compilation_result.proto",
    ],
    protodeps = tf_additional_all_protos() + [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)
