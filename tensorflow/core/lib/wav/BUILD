# Description:
# wav_io targets.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//tensorflow/core:__pkg__"],
    licenses = ["notice"],
)

exports_files(srcs = [
    "wav_io.cc",
    "wav_io.h",
    "wav_io_test.cc",
])

cc_library(
    name = "wav_io",
    srcs = [
        "wav_io.cc",
    ],
    hdrs = ["wav_io.h"],
    deps = [
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base",
    ],
)
