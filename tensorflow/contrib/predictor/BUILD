# `Predictor` classes provide an interface for efficient, repeated inference.

package(default_visibility = ["//tensorflow/contrib/predictor:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "predictor",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":predictor_factories",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "predictor_factories",
    srcs = ["predictor_factories.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":contrib_estimator_predictor",
        ":core_estimator_predictor",
        ":saved_model_predictor",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "base_predictor",
    srcs = ["predictor.py"],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

py_library(
    name = "saved_model_predictor",
    srcs = ["saved_model_predictor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_predictor",
        "//tensorflow/contrib/saved_model:saved_model_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_library(
    name = "core_estimator_predictor",
    srcs = ["core_estimator_predictor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_predictor",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_library(
    name = "contrib_estimator_predictor",
    srcs = ["contrib_estimator_predictor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_predictor",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "testing_common",
    srcs = ["testing_common.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "predictor_pip",
    visibility = ["//visibility:public"],
    deps = [
        ":contrib_estimator_predictor",
        ":core_estimator_predictor",
        ":saved_model_predictor",
    ],
)

py_test(
    name = "saved_model_predictor_test",
    srcs = ["saved_model_predictor_test.py"],
    data = [":test_export_dir"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":saved_model_predictor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "predictor_factories_test",
    srcs = ["predictor_factories_test.py"],
    data = [":test_export_dir"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":predictor_factories",
        ":testing_common",
    ],
)

py_test(
    name = "core_estimator_predictor_test",
    srcs = ["core_estimator_predictor_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":core_estimator_predictor",
        ":testing_common",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "contrib_estimator_predictor_test",
    srcs = ["contrib_estimator_predictor_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":contrib_estimator_predictor",
        ":testing_common",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "test_export_dir",
    srcs = glob(["test_export_dir/**/*"]),
    tags = ["no_pip"],
)
