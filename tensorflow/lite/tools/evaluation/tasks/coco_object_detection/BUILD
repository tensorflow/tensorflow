package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "preprocess_coco_minival",
    srcs = ["preprocess_coco_minival.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/tools/evaluation/proto:evaluation_stages_py"],
)
