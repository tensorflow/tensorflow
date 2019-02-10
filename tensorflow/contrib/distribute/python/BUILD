# Implementation of a prototype TF distributed computation library.

load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("//tensorflow/core:platform/default/distribute.bzl", "distribute_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "distribute_test_lib_pip",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":combinations",
        ":keras_correctness_test_lib",
        ":keras_test_lib",
        ":multi_worker_test_base",
        ":single_loss_example",
        ":strategy_test_lib",
    ],
)

cuda_py_test(
    name = "values_test",
    srcs = ["values_test.py"],
    additional_deps = [
        ":combinations",
        ":mirrored_strategy",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

cuda_py_test(
    name = "input_lib_test",
    srcs = ["input_lib_test.py"],
    additional_deps = [
        ":combinations",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
    ],
)

py_library(
    name = "parameter_server_strategy",
    srcs = ["parameter_server_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
    ],
)

cuda_py_test(
    name = "parameter_server_strategy_test",
    srcs = ["parameter_server_strategy_test.py"],
    additional_deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":parameter_server_strategy",
        ":strategy_test_lib",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:layers",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

py_library(
    name = "one_device_strategy",
    srcs = ["one_device_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:one_device_strategy",
    ],
)

cuda_py_test(
    name = "one_device_strategy_test",
    srcs = ["one_device_strategy_test.py"],
    additional_deps = [
        ":strategy_test_lib",
        ":combinations",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "collective_all_reduce_strategy",
    srcs = ["collective_all_reduce_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
    ],
)

py_library(
    name = "strategy_test_lib",
    srcs = ["strategy_test_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "combinations",
    srcs = ["combinations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mirrored_strategy",
        ":one_device_strategy",
        ":parameter_server_strategy",
        ":tpu_strategy",
        "//tensorflow/contrib/cluster_resolver:cluster_resolver_pip",
        "//tensorflow/contrib/optimizer_v2:training",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras/optimizer_v2",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "combinations_test",
    srcs = ["combinations_test.py"],
    deps = [
        ":combinations",
        "//tensorflow/python/eager:test",
    ],
)

# TODO(priyag): Rename this test to mirrored_strategy_test
cuda_py_test(
    name = "mirrored_strategy_multigpu_test",
    srcs = ["mirrored_strategy_multigpu_test.py"],
    additional_deps = [
        ":combinations",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
    shard_count = 5,
    tags = [
        "guitar",
        "multi_and_single_gpu",
    ],
)

py_library(
    name = "multi_worker_test_base",
    srcs = ["multi_worker_test_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator:estimator_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "step_fn",
    srcs = ["step_fn.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:training",
        "//tensorflow/python/eager:backprop",
    ],
)

py_library(
    name = "tpu_strategy",
    srcs = ["tpu_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":one_device_strategy",
        "//tensorflow/contrib/tpu:tpu_lib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:numpy_dataset",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
    ],
)

cuda_py_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":multi_worker_test_base",
        ":strategy_test_lib",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

distribute_py_test(
    name = "minimize_loss_test",
    srcs = ["minimize_loss_test.py"],
    main = "minimize_loss_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":mirrored_strategy",
        ":single_loss_example",
        "//tensorflow/contrib/tpu:tpu_lib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    additional_deps = [
        ":combinations",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "optimizer_v2_test",
    srcs = ["optimizer_v2_test.py"],
    additional_deps = [
        ":combinations",
        ":single_loss_example",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "estimator_integration_test",
    srcs = ["estimator_integration_test.py"],
    additional_deps = [
        ":combinations",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/contrib/optimizer_v2:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/feature_column",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # http://b/119349471
        "tf_integration_test",
    ],
)

cuda_py_test(
    name = "keras_optimizer_v2_test",
    srcs = ["keras_optimizer_v2_test.py"],
    additional_deps = [
        ":keras_test_lib",
    ],
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # http://b/119349471
        "tf_integration_test",
    ],
)

cuda_py_test(
    name = "estimator_training_test",
    srcs = ["estimator_training_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        ":combinations",
        ":mirrored_strategy",
        ":multi_worker_test_base",
        ":parameter_server_strategy",
        "//third_party/py/numpy",
        "//tensorflow/contrib/optimizer_v2:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/feature_column",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
    shard_count = 48,
    tags = [
        "multi_and_single_gpu",
        # TODO(b/118768923): Re-enable {a,m,t}san test.
        "noasan",
        "nomsan",
        "notsan",
        "no_oss",  # http://b/119349471
    ],
)

py_library(
    name = "single_loss_example",
    srcs = ["single_loss_example.py"],
    deps = [
        ":step_fn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

distribute_py_test(
    name = "step_fn_test",
    srcs = ["step_fn_test.py"],
    main = "step_fn_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        ":single_loss_example",
        "//tensorflow/contrib/tpu:tpu_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "monitor",
    srcs = ["monitor.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "monitor_test",
    srcs = ["monitor_test.py"],
    additional_deps = [
        ":combinations",
        ":monitor",
        ":single_loss_example",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "cross_device_utils_test",
    srcs = ["cross_device_utils_test.py"],
    additional_deps = [
        ":combinations",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

cuda_py_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    additional_deps = [
        ":combinations",
        ":multi_worker_test_base",
        ":mirrored_strategy",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

py_library(
    name = "keras_test_lib",
    srcs = [
        "keras_backward_compat_test.py",
        "keras_test.py",
    ],
    deps = [
        ":combinations",
        "//tensorflow/contrib/distribute/python:mirrored_strategy",
        "//tensorflow/contrib/distribute/python:tpu_strategy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "keras_test",
    srcs = ["keras_test.py"],
    full_precision = True,
    main = "keras_test.py",
    shard_count = 32,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_test_lib",
    ],
)

# TODO(b/121200287): Remove this in 2.0
distribute_py_test(
    name = "keras_backward_compat_test",
    srcs = ["keras_backward_compat_test.py"],
    full_precision = True,
    main = "keras_backward_compat_test.py",
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_test_lib",
    ],
)

py_library(
    name = "keras_correctness_test_lib",
    srcs = [
        "keras_correctness_test_base.py",
        "keras_dnn_correctness_test.py",
        "keras_embedding_model_correctness_test.py",
        "keras_image_model_correctness_test.py",
        "keras_lstm_model_correctness_test.py",
        "keras_stateful_lstm_model_correctness_test.py",
    ],
    deps = [
        ":combinations",
        "//tensorflow/contrib/distribute/python:mirrored_strategy",
        "//tensorflow/contrib/distribute/python:tpu_strategy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "keras_dnn_correctness_test",
    size = "medium",
    srcs = ["keras_dnn_correctness_test.py"],
    full_precision = True,
    main = "keras_dnn_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 19,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_image_model_correctness_test",
    size = "medium",
    srcs = ["keras_image_model_correctness_test.py"],
    full_precision = True,
    main = "keras_image_model_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_embedding_model_correctness_test",
    size = "medium",
    srcs = ["keras_embedding_model_correctness_test.py"],
    full_precision = True,
    main = "keras_embedding_model_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_lstm_model_correctness_test",
    size = "medium",
    srcs = ["keras_lstm_model_correctness_test.py"],
    full_precision = True,
    main = "keras_lstm_model_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_stateful_lstm_model_correctness_test",
    size = "medium",
    srcs = ["keras_stateful_lstm_model_correctness_test.py"],
    full_precision = True,
    main = "keras_stateful_lstm_model_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_pip",
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "metrics_v1_test",
    srcs = ["metrics_v1_test.py"],
    main = "metrics_v1_test.py",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":combinations",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    additional_deps = [
        ":combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "checkpoint_utils_test",
    size = "medium",
    srcs = ["checkpoint_utils_test.py"],
    additional_deps = [
        ":combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

tf_xla_py_test(
    name = "checkpointing_test",
    srcs = ["checkpointing_test.py"],
    disabled_backends = [
        # Only makes sense on TPUs
        "cpu",
        "gpu",
        "cpu_ondemand",
    ],
    tags = [
        "no_oss",
    ],
    deps = [
        ":tpu_strategy",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/training/checkpointable:util",
    ],
)
