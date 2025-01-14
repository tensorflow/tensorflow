# Description:
#   HLO Support Utils

# copybara:uncomment load("@rules_python//python:proto.bzl", "py_proto_library")
load("//xla:package_groups.bzl", "xla_package_groups")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

xla_package_groups()

tf_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "hlo_proto_py_pb2",
#     visibility = internal_visibility([":friends"]),
#     deps = [":hlo_proto"],
# )
# copybara:uncomment_end
