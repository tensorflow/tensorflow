licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python/interpreter_wrapper:tensorflow_wrap_interpreter_wrapper",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = ["//tensorflow/lite/python/testdata:interpreter_test_data"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":interpreter",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "tflite_convert",
    srcs = ["tflite_convert.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":tflite_convert_lib"],
)

py_library(
    name = "tflite_convert_lib",
    srcs = ["tflite_convert.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite",
    ],
)

py_library(
    name = "lite",
    srcs = ["lite.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":convert",
        ":convert_saved_model",
        ":interpreter",
        ":lite_constants",
        ":op_hint",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python/keras",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
    ],
)

py_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    data = ["@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "lite_flex_test",
    srcs = ["lite_flex_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        # TODO(b/111881877): Enable in oss after resolving op registry issues.
        "no_oss",
        "no_windows",
    ],
    deps = [
        ":lite",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "lite_constants",
    srcs = ["lite_constants.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":lite_constants",
        "//tensorflow/lite/toco:model_flags_proto_py",
        "//tensorflow/lite/toco:toco_flags_proto_py",
        "//tensorflow/lite/toco/python:tensorflow_wrap_toco",
        "//tensorflow/lite/toco/python:toco_from_protos",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "convert_test",
    srcs = ["convert_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convert",
        ":interpreter",
        ":op_hint",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
    ],
)

py_library(
    name = "convert_saved_model",
    srcs = ["convert_saved_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":convert",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:platform",
        "//tensorflow/python/saved_model",
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
    tags = [
        "no_windows",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_saved_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python/keras",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model",
    ],
)
