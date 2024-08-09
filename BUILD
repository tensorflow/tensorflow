exports_files(glob(["requirements*"]) + [
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])

cc_library(
    name = "tensorflow_cuda",
    srcs = glob(["tensorflow/core/kernels/cuda/*.cc"]),
    deps = [
        "@cuda//:cuda",
        "@cudnn//:cudnn",
    ],
)

cc_binary(
    name = "tensorflow_cuda_binary",
    srcs = glob(["tensorflow/core/kernels/cuda/*.cc"]),
    deps = [
        ":tensorflow_cuda",
        "@tensorflow_core",
    ],
)

