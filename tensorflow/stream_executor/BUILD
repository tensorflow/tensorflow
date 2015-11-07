licenses(["restricted"])

load("/tensorflow/tensorflow", "if_cuda")

cc_library(
    name = "stream_executor",
    srcs = glob(
        [
            "*.cc",
            "lib/*.cc",
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
        "lib/*.h",
        "platform/**/*.h",
    ]),
    data = [
        "//tensorflow/core:cuda",
        "//third_party/gpus/cuda:cublas",
        "//third_party/gpus/cuda:cudnn",
    ],
    linkopts = [
        "-ldl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//third_party/gpus/cuda:cuda_headers",
    ],
    alwayslink = 1,
)
