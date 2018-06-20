package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "estimator_py",
    srcs = ["estimator_lib.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:internal",
    ],
    deps = [
        ":baseline",
        ":boosted_trees",
        ":dnn",
        ":dnn_linear_combined",
        ":estimator",
        ":export",
        ":exporter",
        ":inputs",
        ":keras",
        ":linear",
        ":model_fn",
        ":parsing_utils",
        ":run_config",
        ":training",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gc",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:util",
    ],
)

py_test(
    name = "exporter_test",
    size = "small",
    srcs = ["exporter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":exporter",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "gc",
    srcs = ["gc.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "gc_test",
    size = "small",
    srcs = ["gc_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gc",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "model_fn",
    srcs = ["model_fn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "@six_archive//:six",
    ],
)

py_test(
    name = "model_fn_test",
    size = "small",
    srcs = ["model_fn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        ":model_fn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":exporter",
        ":run_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_test(
    name = "training_test",
    size = "medium",
    srcs = ["training_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":dnn",
        ":estimator",
        ":exporter",
        ":inputs",
        ":run_config",
        ":training",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/feature_column",
    ],
)

py_library(
    name = "run_config",
    srcs = ["run_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "run_config_test",
    size = "small",
    srcs = ["run_config_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":run_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "baseline",
    srcs = ["canned/baseline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        ":optimizers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "@six_archive//:six",
    ],
)

py_test(
    name = "baseline_test",
    size = "medium",
    srcs = ["canned/baseline_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # test flakily times out in asan mode.
        "notsan",  # b/67510291
        "optonly",  # flakily times out in fastbuild
    ],
    deps = [
        ":baseline",
        ":estimator",
        ":export_export",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "@six_archive//:six",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["canned/boosted_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
    ],
)

py_test(
    name = "boosted_trees_test",
    size = "medium",
    srcs = ["canned/boosted_trees_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "optonly",
    ],
    deps = [
        ":boosted_trees",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/feature_column",
    ],
)

py_library(
    name = "dnn",
    srcs = ["canned/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        ":optimizers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:summary",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_testing_utils",
    testonly = 1,
    srcs = ["canned/dnn_testing_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":metric_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "dnn_test",
    size = "medium",
    srcs = ["canned/dnn_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn",
        ":dnn_testing_utils",
        ":export_export",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["canned/dnn_linear_combined.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dnn",
        ":estimator",
        ":head",
        ":linear",
        ":model_fn",
        ":optimizers",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "@six_archive//:six",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "medium",
    srcs = ["canned/dnn_linear_combined_test.py"],
    shard_count = 8,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn_linear_combined",
        ":dnn_testing_utils",
        ":export_export",
        ":linear_testing_utils",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "@six_archive//:six",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/data",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/67510291
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:training",
        "//tensorflow/python/data",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        ":model_fn",
        ":run_config",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:metrics",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/data",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/67510291
    deps = [
        ":estimator",
        ":export_export",
        ":export_output",
        ":model_fn",
        ":numpy_io",
        ":run_config",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver_test_utils",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/data",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "parsing_utils",
    srcs = [
        "canned/parsing_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/feature_column",
        "@six_archive//:six",
    ],
)

py_test(
    name = "parsing_utils_test",
    srcs = ["canned/parsing_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":parsing_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/feature_column",
    ],
)

py_library(
    name = "export_output",
    srcs = ["export/export_output.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/saved_model:signature_def_utils",
        "@six_archive//:six",
    ],
)

py_test(
    name = "export_output_test",
    size = "small",
    srcs = ["export/export_output_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_library(
    name = "export",
    srcs = [
        "export/export_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        ":export_output",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "export_export",
    srcs = [
        "export/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_test(
    name = "export_test",
    size = "small",
    srcs = ["export/export_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        ":export_output",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
    ],
)

py_library(
    name = "head",
    srcs = ["canned/head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        ":metric_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:signature_constants",
        "@six_archive//:six",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["canned/head_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dnn_testing_utils",
        ":head",
        ":metric_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:signature_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs/inputs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_io",
        ":pandas_io",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "linear",
    srcs = ["canned/linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":optimizers",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "@six_archive//:six",
    ],
)

py_library(
    name = "linear_testing_utils",
    testonly = 1,
    srcs = ["canned/linear_testing_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":export_export",
        ":linear",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "@six_archive//:six",
    ],
)

py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["canned/linear_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":linear",
        ":linear_testing_utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "metric_keys",
    srcs = ["canned/metric_keys.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_fn",
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["inputs/numpy_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
    ],
)

py_test(
    name = "numpy_io_test",
    size = "small",
    srcs = ["inputs/numpy_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_io",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["canned/optimizers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:training",
        "@six_archive//:six",
    ],
)

py_test(
    name = "optimizers_test",
    size = "small",
    srcs = ["canned/optimizers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["inputs/pandas_io.py"],
    srcs_version = "PY2AND3",
    deps = [":inputs_queues"],
)

py_test(
    name = "pandas_io_test",
    size = "small",
    srcs = ["inputs/pandas_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pandas_io",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "prediction_keys",
    srcs = ["canned/prediction_keys.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "inputs_queues",
    srcs = [
        "inputs/queues/__init__.py",
        "inputs/queues/feeding_functions.py",
        "inputs/queues/feeding_queue_runner.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "@six_archive//:six",
    ],
)

py_test(
    name = "feeding_functions_test",
    size = "small",
    srcs = [
        "inputs/queues/feeding_functions_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "feeding_queue_runner_test",
    size = "small",
    srcs = ["inputs/queues/feeding_queue_runner_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":export_export",
        ":model_fn",
        ":run_config",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:training_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_test(
    name = "keras_test",
    size = "large",
    srcs = ["keras_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:engine",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_pandas_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect pandas to already be installed on the system, e.g. via
    # `pip install pandas`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_six_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect six to already be installed on the system, e.g. via
    # `pip install six`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow_gpu`
    visibility = ["//visibility:public"],
)
