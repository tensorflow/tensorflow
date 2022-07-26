load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "if_windows")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:stacktrace",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/stream_executor/platform",
    ] + if_static([
        "//tensorflow/core/platform:env_impl",
    ]),
)
