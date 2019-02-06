package(default_visibility = [
    "//tensorflow/lite:__subpackages__",
    "//tensorflow/tools/pip_package:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary")

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = [
        "//tensorflow/contrib/lite:__subpackages__",
        "//tensorflow/lite:__subpackages__",
    ],
)

cc_library(
    name = "toco_python_api",
    srcs = ["toco_python_api.cc"],
    hdrs = ["toco_python_api.h"],
    deps = [
        "//third_party/python_runtime:headers",
        "//tensorflow/core:lib",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:toco_graphviz_dump_options",
        "//tensorflow/lite/toco:toco_port",
        "//tensorflow/lite/toco:toco_tooling",
    ] + select({
        # This is required when running `tflite_convert` from `bazel`.
        # It requires to link with TensorFlow Ops to get the op definitions.
        ":tflite_convert_with_select_tf_ops": [
            "//tensorflow/core:ops",
        ],
        "//conditions:default": [],
    }),
)

tf_py_wrap_cc(
    name = "tensorflow_wrap_toco",
    srcs = ["toco.i"],
    visibility = [
        "//learning/expander/pod/deep_pod/utils:__subpackages__",
        "//research/handwriting/converters/tflite:__subpackages__",
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":toco_python_api",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
    ],
)

py_binary(
    name = "toco_from_protos",
    srcs = ["toco_from_protos.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensorflow_wrap_toco",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "toco_from_protos_test",
    srcs = ["toco_from_protos_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/toco:model_flags_proto_py",
        "//tensorflow/lite/toco:toco_flags_proto_py",
    ],
    data = [
        ":toco_from_protos",
    ],
    tags = [
        "no_oss",
        "no_pip",
    ],
)
