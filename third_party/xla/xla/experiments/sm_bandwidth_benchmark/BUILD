load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("//xla:xla.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "sm_bw_utils",
    hdrs = ["sm_bw_utils.h"],
    deps = [
        "@local_tsl//tsl/platform:logging",
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

xla_cc_test(
    name = "sm_bw_test",
    srcs = ["sm_bw_test.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":sm_bw_kernels",
        ":sm_bw_utils",
        "@com_google_googletest//:gtest_main",
    ] + if_cuda([
        "//xla/tsl/cuda:cudart",
    ]),
)
