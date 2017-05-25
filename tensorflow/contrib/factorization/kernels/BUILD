# OpKernels for data factorization and clustering.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "all_kernels",
    deps = [
        ":clustering_ops",
        ":masked_matmul_ops",
        ":wals_solver_ops",
        "@protobuf//:protobuf_headers",
    ],
)

cc_library(
    name = "wals_solver_ops",
    srcs = ["wals_solver_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "clustering_ops",
    srcs = ["clustering_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "masked_matmul_ops",
    srcs = ["masked_matmul_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
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
