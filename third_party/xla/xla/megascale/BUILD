load("//xla/megascale:package_groups.bzl", "megascale_package_groups")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
        ":__subpackages__",
    ]),
    licenses = ["notice"],
)

megascale_package_groups()

tf_proto_library(
    name = "dcn_topology_proto",
    srcs = ["dcn_topology.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "addresses_proto",
    srcs = ["addresses.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
)
