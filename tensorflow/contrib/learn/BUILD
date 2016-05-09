# Description:
#   Contains TF Learn (aka Scikit Flow) sub-project with high level tensorflow API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

filegroup(
    name = "datasets",
    srcs = glob(["**/*.csv"]),
)

py_library(
    name = "learn",
    srcs = glob([
        "python/learn/**/*.py",
    ]) + [
        "__init__.py",
        "python/__init__.py",
    ],
    data = [":datasets"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:framework"],
)

py_test(
    name = "test_base",
    size = "medium",
    srcs = ["python/learn/tests/test_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_custom_decay",
    size = "small",
    srcs = ["python/learn/tests/test_custom_decay.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_data_feeder",
    size = "small",
    srcs = ["python/learn/tests/test_data_feeder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_early_stopping",
    size = "medium",
    srcs = ["python/learn/tests/test_early_stopping.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_estimators",
    size = "small",
    srcs = ["python/learn/tests/test_estimators.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_grid_search",
    size = "small",
    srcs = ["python/learn/tests/test_grid_search.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_io",
    size = "small",
    srcs = ["python/learn/tests/test_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_multioutput",
    size = "small",
    srcs = ["python/learn/tests/test_multioutput.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_nonlinear",
    size = "medium",
    srcs = ["python/learn/tests/test_nonlinear.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_regression",
    size = "small",
    srcs = ["python/learn/tests/test_regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_saver",
    size = "small",
    srcs = ["python/learn/tests/test_saver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_ops",
    size = "small",
    srcs = ["python/learn/ops/tests/test_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_dropout_ops",
    size = "small",
    srcs = ["python/learn/ops/tests/test_dropout_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_seq2seq_ops",
    size = "small",
    srcs = ["python/learn/ops/tests/test_seq2seq_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_categorical",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/test_categorical.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_categorical_vocabulary",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/test_categorical_vocabulary.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "test_text",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/test_text.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

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
