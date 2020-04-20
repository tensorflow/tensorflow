load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "run_eval_lib",
    srcs = ["run_eval.cc"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools/evaluation:evaluation_delegate_provider",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tensorflow/lite/tools/evaluation/stages:inference_profiler_stage",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_binary(
    name = "run_eval",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":run_eval_lib",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
    ],
)
