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
    name = "axi_support_sysc",
    srcs = [
        "axi_api_sysc_v2.cc",
    ],
    hdrs = [
        "axi_api_v2.h"
    ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator:systemc_integrator",
    ],
)


cc_library(
    name = "axi_support",
    srcs = [
        "axi_api_v2.cc",
    ],
    hdrs = [
        "axi_api_v2.h"
    ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator:systemc_integrator",
    ],
)