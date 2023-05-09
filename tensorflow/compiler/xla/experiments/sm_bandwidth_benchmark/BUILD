load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")

cc_library(
    name = "sm_bw_utils",
    hdrs = ["sm_bw_utils.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]),
    deps = [
        "//tensorflow/tsl/platform:logging",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cuda_library(
    name = "sm_bw_kernels",
    srcs = ["sm_bw_kernels.cu.cc"],
    hdrs = ["sm_bw_kernels.h"],
    deps = [
        ":sm_bw_utils",
    ],
)

cc_test(
    name = "sm_bw_test",
    srcs = ["sm_bw_test.cc"],
    tags = ["requires-gpu-sm80-only"],
    deps = [
        ":sm_bw_kernels",
        ":sm_bw_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
