load("//tensorflow/lite/experimental/support/metadata:build_defs.bzl", "stamp_metadata_parser_version")

package(
    default_visibility = ["//tensorflow/lite/experimental/support:users"],
    licenses = ["notice"],  # Apache 2.0
)

stamp_metadata_parser_version(
    name = "metadata_parser_h",
    srcs = ["metadata_parser.h.template"],
    outs = ["metadata_parser.h"],
)

cc_library(
    name = "metadata_version",
    srcs = ["metadata_version.cc"],
    hdrs = [
        "metadata_version.h",
        ":metadata_parser_h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_cc",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/tools:logging",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)
