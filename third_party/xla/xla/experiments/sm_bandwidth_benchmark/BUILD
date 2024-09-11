load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//xla:xla.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "sm_bw_utils",
    hdrs = ["sm_bw_utils.h"],
    tags = [
        "gpu",
        "no_rocm",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:logging",
    ],
)

cuda_library(
    name = "sm_bw_kernels",
    srcs = ["sm_bw_kernels.cu.cc"],
    hdrs = ["sm_bw_kernels.h"],
    tags = ["no_rocm"],
    deps = [
        ":sm_bw_utils",
    ],
)

xla_cc_test(
    name = "sm_bw_test",
    srcs = ["sm_bw_test.cc"],
    tags = [
        "gpu",
        "no_rocm",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":sm_bw_kernels",
        ":sm_bw_utils",
        "//xla/tsl/cuda:cudart",
        "@com_google_googletest//:gtest_main",
    ],
)
