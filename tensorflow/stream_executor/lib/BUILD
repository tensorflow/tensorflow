licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

package(default_visibility = [":friends"])

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "lib",
    srcs = glob(
        [
            "**/*.cc",
        ],
        exclude = [
            "**/*test*",
        ],
    ),
    hdrs = glob(["**/*.h"]),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:ptr_util",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "statusor_test",
    size = "small",
    srcs = ["statusor_test.cc"],
    deps = [
        ":lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "utility_headers",
    hdrs = [
        "ptr_util.h",
    ],
    deps = [
        "//tensorflow/core:ptr_util",
    ],
)
