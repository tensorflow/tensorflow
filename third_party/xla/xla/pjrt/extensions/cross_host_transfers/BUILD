load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_cross_host_transfer_extension",
    srcs = ["pjrt_c_api_cross_host_transfers_extension.cc"],
    hdrs = ["pjrt_c_api_cross_host_transfers_extension.h"],
    deps = [
        "//xla:future",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)
