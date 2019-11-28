package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "//tensorflow:android": [
        "-pie",
    ],
})

cc_binary(
    name = "run_eval",
    srcs = ["run_eval.cc"],
    copts = tflite_copts(),
    linkopts = common_linkopts,
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation:utils",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tensorflow/lite/tools/evaluation/stages:image_classification_stage",
    ],
)
