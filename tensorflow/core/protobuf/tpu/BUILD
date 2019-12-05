load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_proto_library_py",
    "tf_pyclif_proto_library",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_proto_library(
    name = "tpu_embedding_configuration_proto",
    srcs = [
        "tpu_embedding_configuration.proto",
    ],
    cc_api_version = 2,
    protodeps = [
        ":tpu_embedding_output_layout_proto",
        ":optimization_parameters_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "optimization_parameters_proto",
    srcs = [
        "optimization_parameters.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "tpu_embedding_output_layout_proto",
    srcs = [
        "tpu_embedding_output_layout.proto",
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

tf_proto_library(
    name = "dynamic_padding_proto",
    srcs = [
        "dynamic_padding.proto",
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

tf_proto_library_cc(
    name = "compile_metadata_proto",
    srcs = [
        "compile_metadata.proto",
    ],
    cc_api_version = 2,
    protodeps = [
        ":dynamic_padding_proto",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "topology_pyclif",
    proto_lib = ":topology_proto",
    proto_srcfile = "topology.proto",
    visibility = ["//visibility:public"],
)
