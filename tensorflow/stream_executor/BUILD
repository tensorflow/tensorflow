licenses(["restricted"])

load("//tensorflow:tensorflow.bzl", "if_cuda")

cc_library(
    name = "stream_executor",
    srcs = glob(
        [
            "*.cc",
            "lib/*.cc",
            "platform/default/*.cc",
        ],
        exclude = [
            "**/*_test.cc",
        ],
    ) + if_cuda(
        glob([
            "cuda/*.cc",
        ]),
    ),
    hdrs = glob([
        "*.h",
        "cuda/*.h",
        "lib/*.h",
        "lib/gtl/*.h",
        "platform/**/*.h",
    ]),
    data = [
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ],
    linkopts = [
        "-ldl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = 1,
)
