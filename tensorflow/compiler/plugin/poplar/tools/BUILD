load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_cc_binary(
    name = "dump_graphs",
    srcs = ["dump_graphs.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)
