# Description:
#   Contains TF Learn (aka Scikit Flow) sub-project with high level tensorflow API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "learn",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ] + glob(
        ["python/learn/**/*.py"],
        exclude = ["python/learn/**/*_test.py"],
    ),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/input_pipeline:input_pipeline_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/linear_optimizer:sdca_ops_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/session_bundle:exporter",
        "//tensorflow/contrib/session_bundle:gc",
        "//tensorflow/contrib/tensor_forest:client_lib",
        "//tensorflow/contrib/tensor_forest:data_ops_py",
        "//tensorflow/contrib/tensor_forest:eval_metrics",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
        "//tensorflow/python:rnn",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_test(
    name = "data_feeder_test",
    size = "small",
    srcs = ["python/learn/learn_io/data_feeder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "feeding_functions_test",
    size = "small",
    srcs = [
        "python/learn/tests/dataframe/feeding_functions_test.py",
    ],
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
    size = "medium",
    srcs = ["python/learn/tests/dataframe/dataframe_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "estimator_utils_test",
    size = "medium",
    srcs = ["python/learn/tests/dataframe/estimator_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_for_generated_wrappers",
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
        ":learn",
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
    name = "sparsify_densify_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/sparsify_densify_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "tensorflow_dataframe_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/tensorflow_dataframe_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "arithmetic_transform_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/arithmetic_transform_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "unary_transform_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/unary_transform_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "binary_transform_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/binary_transform_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "boolean_mask_test",
    size = "small",
    srcs = ["python/learn/tests/dataframe/boolean_mask_test.py"],
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
    srcs = ["python/learn/estimators/estimators_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "metric_spec_test",
    size = "small",
    srcs = ["python/learn/metric_spec_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "experiment_test",
    size = "medium",
    srcs = ["python/learn/experiment_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "graph_actions_test",
    size = "small",
    srcs = ["python/learn/graph_actions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resources",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "learn_runner_test",
    size = "small",
    srcs = ["python/learn/learn_runner_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "monitors_test",
    size = "small",
    srcs = ["python/learn/monitors_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "run_config_test",
    size = "small",
    srcs = [
        "python/learn/estimators/run_config_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
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
    size = "medium",
    srcs = ["python/learn/estimators/estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:util",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_test(
    name = "logistic_regressor_test",
    size = "small",
    srcs = ["python/learn/estimators/logistic_regressor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "large",
    srcs = ["python/learn/estimators/dnn_linear_combined_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["python/learn/estimators/head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dnn_test",
    size = "medium",
    srcs = ["python/learn/estimators/dnn_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "dnn_linear_combined_benchmark_test",
    size = "medium",
    srcs = ["python/learn/estimators/dnn_linear_combined_benchmark_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "guitar",
        "local",
        "manual",
        "notap",
    ],
    visibility = [
        "//learning/brain/google/guitar:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "kmeans_test",
    size = "medium",
    srcs = ["python/learn/estimators/kmeans_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_benchmark",
    ],
)

py_test(
    name = "dnn_benchmark_test",
    size = "medium",
    srcs = ["python/learn/estimators/dnn_benchmark_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "guitar",
        "local",
        "manual",
        "notap",
    ],
    visibility = [
        "//learning/brain/google/guitar:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "random_forest_test",
    size = "medium",
    srcs = ["python/learn/estimators/random_forest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "dynamic_rnn_estimator_test",
    size = "medium",
    srcs = ["python/learn/estimators/dynamic_rnn_estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["python/learn/estimators/linear_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "composable_model_test",
    size = "medium",
    srcs = ["python/learn/estimators/composable_model_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:state_ops",
    ],
)

py_test(
    name = "svm_test",
    size = "medium",
    srcs = ["python/learn/estimators/svm_test.py"],
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
    srcs = ["python/learn/grid_search_test.py"],
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
    srcs = ["python/learn/learn_io/io_test.py"],
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
    srcs = ["python/learn/estimators/multioutput_test.py"],
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
    srcs = ["python/learn/estimators/nonlinear_test.py"],
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
    srcs = ["python/learn/estimators/regression_test.py"],
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
    srcs = ["python/learn/ops/ops_test.py"],
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
    srcs = ["python/learn/ops/seq2seq_ops_test.py"],
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
    name = "graph_io_test",
    size = "small",
    srcs = ["python/learn/learn_io/graph_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "numpy_io_test",
    size = "small",
    srcs = ["python/learn/learn_io/numpy_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "pandas_io_test",
    size = "small",
    srcs = ["python/learn/learn_io/pandas_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "export_test",
    size = "small",
    srcs = ["python/learn/utils/export_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",  # http://b/31032996
    ],
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "gc_test",
    size = "small",
    srcs = ["python/learn/utils/gc_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "saved_model_export_utils_test",
    size = "small",
    srcs = ["python/learn/utils/saved_model_export_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
    ],
)

py_test(
    name = "stability_test",
    size = "small",
    srcs = ["python/learn/estimators/stability_test.py"],
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
