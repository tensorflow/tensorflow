load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "preprocess_coco_minival",
    srcs = ["preprocess_coco_minival.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/tools/evaluation/proto:evaluation_stages_py"],
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
        "//tensorflow/lite/tools/evaluation:utils",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tensorflow/lite/tools/evaluation/stages:object_detection_stage",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
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
