# Description:
#   Contains ops for evaluation metrics and summary statistics.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "metrics_py",
    srcs = [
        "__init__.py",
        "python/metrics/__init__.py",
        "python/metrics/classification.py",
        "python/ops/confusion_matrix_ops.py",
        "python/ops/histogram_ops.py",
        "python/ops/metric_ops.py",
        "python/ops/set_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sets",
    ],
)

py_test(
    name = "classification_test",
    srcs = ["python/metrics/classification_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "histogram_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/histogram_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "metric_ops_test",
    srcs = ["python/ops/metric_ops_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":metrics_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
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
)
