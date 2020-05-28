package(
    default_visibility = ["//tensorflow/lite/experimental/support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metadata_version",
    srcs = ["metadata_version.cc"],
    hdrs = ["metadata_version.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_cc",
        "//tensorflow/lite/tools:logging",
        "@flatbuffers",
    ],
)
