# Description:
#   Tools for manipulating TensorFlow graphs.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "py_binary")

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "tools_pip",
    deps = [
        ":freeze_graph",
        ":import_pb_to_tensorboard",
        ":inspect_checkpoint",
        ":optimize_for_inference",
        ":print_selective_registration_header",
        ":saved_model_cli",
        ":saved_model_utils",
        ":strip_unused",
        # The following py_library are needed because
        # py_binary may not depend on them when --define=no_tensorflow_py_deps=true
        # is specified. See https://github.com/tensorflow/tensorflow/issues/22390
        ":freeze_graph_lib",
        ":optimize_for_inference_lib",
        ":selective_registration_header_lib",
        ":strip_unused_lib",
        # Include the TF upgrade script to users can run it directly after install TF
        "//tensorflow/tools/compatibility:tf_upgrade_v2",
    ],
)

py_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "saved_model_utils_test",
    size = "small",
    srcs = ["saved_model_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    visibility = ["//visibility:private"],
    deps = [
        ":saved_model_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/saved_model",
    ],
)

py_library(
    name = "freeze_graph_lib",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:no_contrib",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/saved_model:loader",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "freeze_graph",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":freeze_graph_lib",
    ],
)

py_binary(
    name = "import_pb_to_tensorboard",
    srcs = ["import_pb_to_tensorboard.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
)

py_test(
    name = "freeze_graph_test",
    size = "small",
    srcs = ["freeze_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":freeze_graph",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_binary(
    name = "inspect_checkpoint",
    srcs = ["inspect_checkpoint.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_library(
    name = "strip_unused_lib",
    srcs = ["strip_unused_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "component_api_helper",
    srcs = ["component_api_helper.py"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "strip_unused",
    srcs = ["strip_unused.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "@six_archive//:six",
    ],
)

py_test(
    name = "strip_unused_test",
    size = "small",
    srcs = ["strip_unused_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notap"],
    deps = [
        ":strip_unused_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "optimize_for_inference_lib",
    srcs = ["optimize_for_inference_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "optimize_for_inference",
    srcs = ["optimize_for_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":optimize_for_inference_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "@six_archive//:six",
    ],
)

py_test(
    name = "optimize_for_inference_test",
    size = "small",
    srcs = ["optimize_for_inference_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":optimize_for_inference_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "selective_registration_header_lib",
    srcs = ["selective_registration_header_lib.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "print_selective_registration_header",
    srcs = ["print_selective_registration_header.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":selective_registration_header_lib",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "print_selective_registration_header_test",
    srcs = ["print_selective_registration_header_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":selective_registration_header_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "saved_model_cli",
    srcs = ["saved_model_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_utils",
        "//tensorflow/python",
        "//tensorflow/python/debug:local_cli_wrapper",
    ],
)

py_test(
    name = "saved_model_cli_test",
    srcs = ["saved_model_cli_test.py"],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no-internal-py3",
    ],
    deps = [
        ":saved_model_cli",
        "//tensorflow/core:protos_all_py",
    ],
)
