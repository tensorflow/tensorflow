load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "task_executor",
    srcs = ["task_executor.cc"],
    hdrs = ["task_executor.h"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools/evaluation:evaluation_delegate_provider",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "task_executor_main",
    srcs = ["task_executor_main.cc"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":task_executor",
        "//tensorflow/lite/tools:logging",
        "@com_google_absl//absl/types:optional",
    ],
)
