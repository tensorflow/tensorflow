load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(
        ["//xla:__subpackages__"],
    ),
    licenses = ["notice"],
)

exports_files(
    ["xprof_loader.h"],
    visibility = internal_visibility(
        ["//xla/tools/platform:__subpackages__"],
    ),
)

cc_library(
    name = "xprof_loader",
    hdrs = ["xprof_loader.h"],
    compatible_with = [],
    textual_hdrs = ["xprof_loader.h"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
    ] + if_google(
        ["//xla/tools/platform/google:xprof_loader"],
        ["//xla/tools/platform/default:xprof_loader"],
    ),
)
