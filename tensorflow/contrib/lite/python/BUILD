licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "interpreter_test_data",
    srcs = glob(["**/testdata/*"]),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/lite/python/interpreter_wrapper:tensorflow_wrap_interpreter_wrapper",
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [":interpreter_test_data"],
    srcs_version = "PY2AND3",
    tags = ["no_oss"],
    deps = [
        ":interpreter",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "lite",
    srcs = ["lite.py"],
    # data = [
    #     "//tensorflow/contrib/lite/toco/python:toco_from_protos",
    # ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":op_hint",
        "//tensorflow/contrib/lite/toco:model_flags_proto_py",
        "//tensorflow/contrib/lite/toco:toco_flags_proto_py",
        "//tensorflow/contrib/lite/toco/python:tensorflow_wrap_toco",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no-internal-py3",
        "no_oss",
    ],
    deps = [
        ":lite",
        ":op_hint",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
    ],
)

py_binary(
    name = "convert_saved_model",
    srcs = ["convert_saved_model.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite",
        "//tensorflow/contrib/saved_model:saved_model_py",
        "//tensorflow/python:graph_util",
        "//tensorflow/python/tools:freeze_graph_lib",
    ],
)

py_binary(
    name = "create_custom_op",
    srcs = ["create_custom_op.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "convert_saved_model_test",
    srcs = ["convert_saved_model_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_saved_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "tf_lite_py_pip",
    deps = [
        ":convert_saved_model",
    ],
)
