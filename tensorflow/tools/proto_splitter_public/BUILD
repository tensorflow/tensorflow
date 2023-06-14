# Description:
#   Fake exports until ProtoSplitter is exported to OSS.
#   (original description) Utilities for splitting and joining large protos > 2GB.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "internal_api",
    srcs = ["internal_api.cc"],
    hdrs = ["internal_api.h"],
    deps = [
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status",
    ],
)
