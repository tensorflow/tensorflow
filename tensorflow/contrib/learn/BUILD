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
    name = "base_test",
    size = "medium",
    srcs = ["python/learn/tests/base_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "data_feeder_test",
    size = "small",
    srcs = ["python/learn/tests/data_feeder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "feeding_queue_runner_test",
    size = "small",
    srcs = [
        "python/learn/tests/dataframe/feeding_queue_runner_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dataframe_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/dataframe_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "series_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/series_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "transform_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/transform_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "reader_source_test",
    size = "small",
    srcs = [
        "python/learn/dataframe/transforms/reader_source.py",
        "python/learn/tests/dataframe/reader_source_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "csv_parser_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/csv_parser_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "example_parser_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/example_parser_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "in_memory_source_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/in_memory_source_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "batch_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/batch_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "early_stopping_test",
    size = "medium",
    srcs = ["python/learn/tests/early_stopping_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "estimators_test",
    size = "small",
    srcs = ["python/learn/tests/estimators_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "tensor_signature_test",
    srcs = ["python/learn/estimators/tensor_signature_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "estimator_test",
    size = "small",
    srcs = ["python/learn/estimators/estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "classifier_test",
    size = "small",
    srcs = ["python/learn/estimators/classifier_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "medium",
    srcs = ["python/learn/estimators/dnn_linear_combined_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dnn_test",
    size = "small",
    srcs = ["python/learn/estimators/dnn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "linear_test",
    size = "small",
    srcs = ["python/learn/estimators/linear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "grid_search_test",
    size = "small",
    srcs = ["python/learn/tests/grid_search_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "io_test",
    size = "small",
    srcs = ["python/learn/tests/io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "multioutput_test",
    size = "small",
    srcs = ["python/learn/tests/multioutput_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "nonlinear_test",
    size = "medium",
    srcs = ["python/learn/tests/nonlinear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "regression_test",
    size = "small",
    srcs = ["python/learn/tests/regression_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "saver_test",
    size = "small",
    srcs = ["python/learn/tests/saver_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/learn/ops/tests/ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dropout_ops_test",
    size = "small",
    srcs = ["python/learn/ops/tests/dropout_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "seq2seq_ops_test",
    size = "small",
    srcs = ["python/learn/ops/tests/seq2seq_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "categorical_test",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/categorical_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "categorical_vocabulary_test",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/categorical_vocabulary_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/text_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "checkpoints_test",
    size = "small",
    srcs = ["python/learn/utils/checkpoints_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "graph_io_test",
    size = "small",
    srcs = ["python/learn/io/graph_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_binary(
    name = "inspect_checkpoint",
    srcs = [
        "python/learn/utils/inspect_checkpoint.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
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
