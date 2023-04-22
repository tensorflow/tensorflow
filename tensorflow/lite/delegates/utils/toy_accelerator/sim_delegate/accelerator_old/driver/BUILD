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
    name = "toy_driver",
    srcs = [
        "systemc_binding.h",
        "acc_container.h",
        "add_driver.h",
        ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_tflite/axi_support:axi_support_sysc",
        "//tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator:systemc_integrator",
        "//tensorflow/lite/delegates/utils/secda_tflite/toy_accelerator/sim_delegate/accelerator:toy_accelerator",
    ],
)
