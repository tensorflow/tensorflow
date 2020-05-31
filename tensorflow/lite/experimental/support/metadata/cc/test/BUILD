package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "metadata_version_test",
    srcs = ["metadata_version_test.cc"],
    deps = [
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_cc",
        "//tensorflow/lite/experimental/support/metadata/cc:metadata_version",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
