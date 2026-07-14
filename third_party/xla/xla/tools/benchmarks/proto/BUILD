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
    name = "benchmark_config_proto",
    srcs = ["benchmark_config.proto"],
)
