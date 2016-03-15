# Description:
#   contains Scikit Flow sub-project with high level tensorflow API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "skflow",
    srcs = glob([
        "python/skflow/**/*.py",
    ]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
    ],
)

py_test(
    name = "test_base",
    srcs = ["python/skflow/tests/test_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_custom_decay",
    srcs = ["python/skflow/tests/test_custom_decay.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_data_feeder",
    srcs = ["python/skflow/tests/test_data_feeder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_estimators",
    srcs = ["python/skflow/tests/test_estimators.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_grid_search",
    srcs = ["python/skflow/tests/test_grid_search.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_io",
    srcs = ["python/skflow/tests/test_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_multioutput",
    srcs = ["python/skflow/tests/test_multioutput.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_nonlinear",
    srcs = ["python/skflow/tests/test_nonlinear.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_regression",
    srcs = ["python/skflow/tests/test_regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_saver",
    srcs = ["python/skflow/tests/test_saver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_ops",
    srcs = ["python/skflow/ops/tests/test_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_dropout_ops",
    srcs = ["python/skflow/ops/tests/test_dropout_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_seq2seq_ops",
    srcs = ["python/skflow/ops/tests/test_seq2seq_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_categorical",
    srcs = ["python/skflow/preprocessing/tests/test_categorical.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_categorical_vocabulary",
    srcs = ["python/skflow/preprocessing/tests/test_categorical_vocabulary.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
)

py_test(
    name = "test_text",
    srcs = ["python/skflow/preprocessing/tests/test_text.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
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

