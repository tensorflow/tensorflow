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
    name = "accelerator",
    srcs = [
        "vta.cc",
        ],
    hdrs = [
        "hw_spec_const.h",
        "hw_spec.h",
        "vta.h",
        "fetch.h",
        "load.h",
        "compute.h",
        "store.h",
        "ap_mem_if.h",
        "ap_sc_ext.h",
        "AXI4_if.h",
        "hls_bus_if.h",
    ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/vta_delegate/secda_tflite/accelerator/testbench:profiler",
    ],
)

