package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "estimator_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dnn",
        ":extenders",
        ":head",
        ":multi_head",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "dnn",
    srcs = ["python/estimator/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:nn",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn",
    ],
)

py_test(
    name = "dnn_test",
    size = "small",
    srcs = ["python/estimator/dnn_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dnn",
        ":head",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:dnn_testing_utils",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "extenders",
    srcs = [
        "python/estimator/extenders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:util",
    ],
)

py_test(
    name = "extenders_test",
    size = "small",
    srcs = ["python/estimator/extenders_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":extenders",
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:linear",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/ops/losses",
    ],
)

py_test(
    name = "head_test",
    size = "small",
    srcs = ["python/estimator/head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/saved_model:signature_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "multi_head",
    srcs = [
        "python/estimator/multi_head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/saved_model:signature_constants",
        "@six_archive//:six",
    ],
)

py_test(
    name = "multi_head_test",
    size = "small",
    srcs = ["python/estimator/multi_head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        ":multi_head",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/saved_model:signature_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
