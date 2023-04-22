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
    name = "toy_accelerator",
    srcs = [
        "acc.sc.cc",
        ],
    hdrs = [
        "acc.sc.h",
        "control.sc.h",
        "data_read.sc.h",
        "data_write.sc.h",
        "pe_add.sc.h",
    ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler:profiler",
        "//tensorflow/lite/delegates/utils/secda_tflite/ap_sysc:ap_sysc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator:systemc_integrator",
        
    ],
)

