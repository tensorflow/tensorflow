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
    name = "driver",
    srcs = [
        "gen_instructions.h",
        "systemc_binding.h",
        "acc_container.h",
        "vta_gemm_driver.h",
        ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/vta_sim_delegate/accelerator:accelerator",
        "//tensorflow/lite/delegates/utils/secda_tflite:secda_tflite_sim",
    ],
)

