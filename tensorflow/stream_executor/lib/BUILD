load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "if_windows")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

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
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)
