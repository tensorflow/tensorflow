# copybara:uncomment load("@rules_python//python:proto.bzl", "py_proto_library")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    protodeps = [
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/stream_executor:device_description_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "compile_options_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":compile_options_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "executable_metadata_proto",
    srcs = ["executable_metadata.proto"],
    protodeps = [
        "//xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)
