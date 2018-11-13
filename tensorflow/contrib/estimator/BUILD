package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
# PLACEHOLDER PIP REQUIREMENTS

py_library(
    name = "estimator_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees",
        ":dnn",
        ":dnn_with_layer_annotations",
        ":early_stopping",
        ":expect_tensorflow_estimator_installed",
        ":export",
        ":exporter",
        ":extenders",
        ":head",
        ":hooks",
        ":linear",
        ":logit_fns",
        ":multi_head",
        ":replicate_model_fn",
        ":rnn",
        ":saved_model_estimator",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["python/estimator/boosted_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:boosted_trees",
    ],
)

py_library(
    name = "dnn",
    srcs = ["python/estimator/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn",
    ],
)

py_library(
    name = "dnn_with_layer_annotations",
    srcs = ["python/estimator/dnn_with_layer_annotations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:optimizers",
    ],
)

py_library(
    name = "extenders",
    srcs = [
        "python/estimator/extenders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "export",
    srcs = [
        "python/estimator/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_library(
    name = "exporter",
    srcs = [
        "python/estimator/exporter.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:exporter",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
    ],
)

py_library(
    name = "hooks",
    srcs = [
        "python/estimator/hooks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "linear",
    srcs = ["python/estimator/linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:linear",
    ],
)

py_library(
    name = "logit_fns",
    srcs = [
        "python/estimator/logit_fns.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:dnn",
        "//tensorflow/python/estimator:linear",
    ],
)

py_library(
    name = "multi_head",
    srcs = [
        "python/estimator/multi_head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "@six_archive//:six",
    ],
)

py_library(
    name = "replicate_model_fn",
    srcs = [
        "python/estimator/replicate_model_fn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "rnn",
    srcs = ["python/estimator/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":extenders",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/contrib/feature_column:feature_column_py",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:optimizers",
        "@six_archive//:six",
    ],
)

py_library(
    name = "early_stopping",
    srcs = ["python/estimator/early_stopping.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
    ],
)

py_library(
    name = "saved_model_estimator",
    srcs = ["python/estimator/saved_model_estimator.py"],
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":export",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:export",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_library(
    name = "expect_tensorflow_estimator_installed",
    # This is a dummy rule used as a dependency in open-source.
    # We expect tensorflow_estimator to already be installed.
    visibility = ["//visibility:public"],
)
