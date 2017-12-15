# Description:
#   Contains TF Learn (aka Scikit Flow) sub-project with high level tensorflow API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//tensorflow:internal",
])

load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")

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
    # This library should not depend on sklearn, even though some of the code
    # refers to it. (The code handles the presence of sklearn conditionally.)
    deps = [
        "//tensorflow/contrib/factorization:factorization_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/input_pipeline:input_pipeline_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/linear_optimizer:sdca_ops_py",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/meta_graph_transform",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/contrib/session_bundle:exporter",
        "//tensorflow/contrib/session_bundle:gc",
        "//tensorflow/contrib/tpu:tpu_estimator",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/core:protos_all_py",
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
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:resources",
        "//tensorflow/python:rnn",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:inputs",
        "//tensorflow/python/estimator:inputs_queues",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:pandas_io",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

# Exposes constants without having to build the entire :learn target.
py_library(
    name = "estimator_constants_py",
    srcs = [
        "python/learn/estimators/constants.py",
        "python/learn/estimators/prediction_key.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "data_feeder_test",
    size = "small",
    srcs = ["python/learn/learn_io/data_feeder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "estimators_test",
    size = "small",
    srcs = ["python/learn/estimators/estimators_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "metric_spec_test",
    size = "small",
    srcs = ["python/learn/metric_spec_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "experiment_test",
    size = "medium",
    srcs = ["python/learn/experiment_test.py"],
    additional_deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator",
    ],
)

py_test(
    name = "graph_actions_test",
    size = "small",
    srcs = ["python/learn/graph_actions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:training",
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
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python/estimator:run_config",
    ],
)

py_test(
    name = "monitors_test",
    size = "small",
    srcs = ["python/learn/monitors_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator",
    ],
)

py_test(
    name = "run_config_test",
    size = "small",
    srcs = ["python/learn/estimators/run_config_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:run_config",
    ],
)

py_test(
    name = "tensor_signature_test",
    srcs = ["python/learn/estimators/tensor_signature_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
    ],
)

py_test(
    name = "estimator_test",
    size = "medium",
    srcs = ["python/learn/estimators/estimator_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:protos_all_py",
        "//tensorflow/python:session",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "estimator_input_test",
    size = "medium",
    srcs = ["python/learn/estimators/estimator_input_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "logistic_regressor_test",
    size = "small",
    srcs = ["python/learn/estimators/logistic_regressor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "large",
    srcs = ["python/learn/estimators/dnn_linear_combined_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["no_oss"],  # flaky b/70524820
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["python/learn/estimators/head_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["noasan"],  # times out b/63678675
    deps = [
        ":learn",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
        "@six_archive//:six",
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
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "kmeans_test",
    size = "medium",
    srcs = ["python/learn/estimators/kmeans_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "dynamic_rnn_estimator_test",
    size = "medium",
    srcs = ["python/learn/estimators/dynamic_rnn_estimator_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "state_saving_rnn_estimator_test",
    size = "medium",
    srcs = ["python/learn/estimators/state_saving_rnn_estimator_test.py"],
    srcs_version = "PY2AND3",
    tags = ["noasan"],
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["python/learn/estimators/linear_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/linear_optimizer:sdca_ops_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "debug_test",
    size = "medium",
    srcs = ["python/learn/estimators/debug_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "composable_model_test",
    size = "medium",
    srcs = ["python/learn/estimators/composable_model_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
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
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
    ],
)

py_test(
    name = "grid_search_test",
    size = "small",
    srcs = ["python/learn/grid_search_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "io_test",
    size = "small",
    srcs = ["python/learn/learn_io/io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "model_fn_test",
    size = "small",
    srcs = ["python/learn/estimators/model_fn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/saved_model:signature_constants",
        "@six_archive//:six",
    ],
)

py_test(
    name = "multioutput_test",
    size = "small",
    srcs = ["python/learn/estimators/multioutput_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "nonlinear_test",
    size = "medium",
    srcs = ["python/learn/estimators/nonlinear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:random_seed",
    ],
)

py_test(
    name = "regression_test",
    size = "small",
    srcs = ["python/learn/estimators/regression_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "rnn_common_test",
    size = "medium",
    srcs = ["python/learn/estimators/rnn_common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:session",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/learn/ops/ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "seq2seq_ops_test",
    size = "small",
    srcs = ["python/learn/ops/seq2seq_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:rnn_cell",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "categorical_test",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/categorical_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "categorical_vocabulary_test",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/categorical_vocabulary_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["python/learn/preprocessing/tests/text_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "graph_io_test",
    size = "small",
    srcs = ["python/learn/learn_io/graph_io_test.py"],
    additional_deps = [
        ":learn",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
    grpc_enabled = True,
)

py_test(
    name = "numpy_io_test",
    size = "small",
    srcs = ["python/learn/learn_io/numpy_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "pandas_io_test",
    size = "small",
    srcs = ["python/learn/learn_io/pandas_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "export_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/learn/utils/export_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",  # http://b/31032996
        "notap",  # TODO(b/37950026): Test is flaky
    ],
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/session_bundle:exporter",
        "//tensorflow/contrib/session_bundle:manifest_proto_py_pb2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "gc_test",
    size = "small",
    srcs = ["python/learn/utils/gc_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "saved_model_export_utils_test",
    size = "small",
    srcs = ["python/learn/utils/saved_model_export_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
    ],
)

py_test(
    name = "input_fn_utils_test",
    size = "small",
    srcs = ["python/learn/utils/input_fn_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
    ],
)

py_test(
    name = "stability_test",
    size = "small",
    srcs = ["python/learn/estimators/stability_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":learn",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
    ],
)

py_binary(
    name = "inspect_checkpoint",
    srcs = ["python/learn/utils/inspect_checkpoint.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:platform",
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
