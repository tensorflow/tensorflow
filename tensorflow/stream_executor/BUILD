licenses(["restricted"])

load("//tensorflow/tensorflow", "if_cuda")
load("//third_party/gpus/cuda/platform", "cuda_sdk_version")
load("//third_party/gpus/cuda/platform", "cudnn_sdk_version")

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
        "//third_party/gpus/cuda:cublas",
        "//third_party/gpus/cuda:cudnn",
        "//third_party/gpus/cuda:cufft",
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
    defines = [
        "CUDA_TOOLKIT_VERSION={}".format(cuda_sdk_version()),
        "CUDNN_VERSION={}".format(cudnn_sdk_version())
    ],
)
