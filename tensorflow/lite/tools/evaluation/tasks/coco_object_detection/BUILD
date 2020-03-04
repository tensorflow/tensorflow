package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

py_binary(
    name = "preprocess_coco_minival",
    srcs = ["preprocess_coco_minival.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/tools/evaluation/proto:evaluation_stages_py"],
)

cc_binary(
    name = "run_eval",
    srcs = ["run_eval.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools/evaluation:evaluation_stage",
        "//tensorflow/lite/tools/evaluation:utils",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//tensorflow/lite/tools/evaluation/stages:object_detection_stage",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
