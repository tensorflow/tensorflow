load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = ["-Wall"] + tflite_copts()


cc_library(
    name = "profiler",
    srcs = [
        "profiler.cc",
        ],
    hdrs = [
        "profiler.h"
    ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
    ],
)


cc_library(
    name = "testbench",
    srcs = [
        "gen_instructions.h",
        "systemc_integrator.h",
        "conv2d_driver.h",
        "vta_gemm_driver.h",
        "mm_helper.h",
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/vta_delegate/secda_tflite/accelerator/testbench:vta_driver",
        "@systemc//:systemc",
    ],
)

cc_library(
    name = "vta_driver",
    srcs = [
        "vta_driver.h",
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/vta_delegate/secda_tflite/accelerator:accelerator",
        "@systemc//:systemc",
    ],
)

