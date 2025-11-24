load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/tools/platform:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "xprof_loader",
    srcs = ["xprof_loader.cc"],
    hdrs = ["//xla/tools/platform:xprof_loader.h"],
    compatible_with = [],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
