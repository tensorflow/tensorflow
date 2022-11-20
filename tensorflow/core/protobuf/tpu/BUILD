load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "tpu_embedding_configuration_proto",
    srcs = [
        "tpu_embedding_configuration.proto",
    ],
    cc_api_version = 2,
    protodeps = [
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
    protodeps = ["//tensorflow/compiler/xla/service:hlo_proto"],
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

tf_proto_library(
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

tf_proto_library(
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

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tpu_embedding_configuration_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":tpu_embedding_configuration_proto"],
# )
#
# py_proto_library(
#     name = "optimization_parameters_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":optimization_parameters_proto"],
# )
#
# py_proto_library(
#     name = "topology_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":topology_proto"],
# )
#
# py_proto_library(
#     name = "dynamic_padding_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":dynamic_padding_proto"],
# )
#
# py_proto_library(
#     name = "compilation_result_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":compilation_result_proto"],
# )
#
# py_proto_library(
#     name = "compile_metadata_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":compile_metadata_proto"],
# )
# copybara:uncomment_end
