# OpKernels for data factorization and clustering.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

cc_library(
    name = "all_kernels",
    deps = [
        ":clustering_ops",
        ":wals_solver_ops",
        "@protobuf//:protobuf",
    ],
)

cc_library(
    name = "wals_solver_ops",
    srcs = ["wals_solver_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "clustering_ops",
    srcs = ["clustering_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)

cc_test(
    name = "clustering_ops_test",
    srcs = ["clustering_ops_test.cc"],
    deps = [
        ":clustering_ops",
        "//tensorflow/contrib/factorization:clustering_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
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
