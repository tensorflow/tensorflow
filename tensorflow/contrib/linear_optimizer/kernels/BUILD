# Libraries and Kernels for Linear Optimization.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "loss_updaters",
    hdrs = [
        "hinge-loss.h",
        "logistic-loss.h",
        "loss.h",
        "squared-loss.h",
    ],
    deps = [
        "//google/protobuf",
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "loss_test",
    size = "small",
    srcs = ["loss_test.cc"],
    deps = [
        ":loss_updaters",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "resources",
    srcs = ["resources.cc"],
    hdrs = ["resources.h"],
    deps = [
        "//google/protobuf",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "resources_test",
    size = "small",
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
        ":loss_updaters",
        ":resources",
        "//google/protobuf",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//third_party/eigen3",
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
