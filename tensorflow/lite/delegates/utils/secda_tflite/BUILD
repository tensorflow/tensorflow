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
    name = "secda_tflite",
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler:profiler",
        "//tensorflow/lite/delegates/utils/secda_tflite/ap_sysc:ap_sysc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator:systemc_integrator",
        "//tensorflow/lite/delegates/utils/secda_tflite/threading_utils:threading_utils",
        "//tensorflow/lite/delegates/utils/secda_tflite/axi_support:axi_support",

    ],
)

cc_library(
    name = "secda_tflite_sim",
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler:profiler",
        "//tensorflow/lite/delegates/utils/secda_tflite/ap_sysc:ap_sysc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator:systemc_integrator",
        "//tensorflow/lite/delegates/utils/secda_tflite/threading_utils:threading_utils",
        "//tensorflow/lite/delegates/utils/secda_tflite/axi_support:axi_support_sysc",

    ],
)