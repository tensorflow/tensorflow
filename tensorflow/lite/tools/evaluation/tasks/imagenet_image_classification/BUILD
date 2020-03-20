load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_linkopts = tflite_linkopts() + select({
    "//tensorflow:android": [
        "-pie",  # Android 5.0 and later supports only PIE
        "-lm",  # some builtin ops, e.g., tanh, need -lm
        "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
    ],
    "//conditions:default": [],
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
        "//tensorflow/lite/tools/evaluation:evaluation_delegate_provider",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation:utils",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tensorflow/lite/tools/evaluation/stages:image_classification_stage",
    ],
)
