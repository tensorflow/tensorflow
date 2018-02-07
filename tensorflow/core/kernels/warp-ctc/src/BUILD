package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_library",
    "cc_header_only_library",
)

tf_kernel_library(
    name = "src_gpu",
    gpu_srcs = glob(["*.cc"]),
    deps = ["//tensorflow/core/kernels/warp-ctc/include:libgpu", "//tensorflow/core/kernels/warp-ctc/include/detail:libgpu", "//tensorflow/core/kernels/warp-ctc/include/contrib/moderngpu/include:lib_gpu"],
)

cc_library(
    name = "src_cpu",
    srcs = glob(["*.cpp"]),
    deps = ["//tensorflow/core/kernels/warp-ctc/include:libcpu", "//tensorflow/core/kernels/warp-ctc/include/detail:libcpu"],
)
