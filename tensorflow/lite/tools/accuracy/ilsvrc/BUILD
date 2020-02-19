load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "//tensorflow:android": [
        "-pie",
        "-llog",
    ],
})

cc_library(
    name = "imagenet_model_evaluator",
    srcs = ["imagenet_model_evaluator.cc"],
    hdrs = ["imagenet_model_evaluator.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools/accuracy/ilsvrc/default:custom_delegates",
        "//tensorflow/lite/tools/evaluation:utils",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tensorflow/lite/tools/evaluation/stages:image_classification_stage",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "imagenet_accuracy_eval_lib",
    srcs = ["imagenet_accuracy_eval.cc"],
    hdrs = ["imagenet_accuracy_eval.h"],
    copts = tflite_copts(),
    linkopts = common_linkopts,
    deps = [
        ":imagenet_model_evaluator",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/profiling:time",
        "//tensorflow/lite/tools/accuracy:csv_writer",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_binary(
    name = "imagenet_accuracy_eval",
    srcs = ["imagenet_accuracy_eval_main.cc"],
    copts = tflite_copts(),
    linkopts = common_linkopts,
    deps = [
        ":imagenet_accuracy_eval_lib",
        "//tensorflow/lite/tools:command_line_flags",
    ],
)
