# Libraries and Kernels for Linear Optimization.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "losses",
    hdrs = [
        "hinge-loss.h",
        "logistic-loss.h",
        "squared-loss.h",
    ],
    deps = ["//tensorflow/core:lib"],
)

# TODO(katsiapis): Add tests for losses.

cc_library(
    name = "resources",
    srcs = ["resources.cc"],
    hdrs = ["resources.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "resources_test",
    srcs = ["resources_test.cc"],
    deps = [
        ":resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "sdca_ops",
    srcs = ["sdca_ops.cc"],
    deps = [
        ":losses",
        ":resources",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
