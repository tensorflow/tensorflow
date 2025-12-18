load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "dlpack_types",
    srcs = ["dlpack_types.cc"],
    hdrs = ["dlpack_types.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@dlpack",
    ],
)

cc_library(
    name = "strides",
    srcs = ["strides.cc"],
    hdrs = ["strides.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/types:span",
    ],
)
