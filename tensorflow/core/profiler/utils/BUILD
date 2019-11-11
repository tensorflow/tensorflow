load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
    ],
)

tf_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)
