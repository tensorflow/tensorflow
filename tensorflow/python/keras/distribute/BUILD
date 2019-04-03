# Description:
#   keras/distribute package is intended to serve as the centralized place for things
#   related to dist-strat used by Keras..
load("//tensorflow/core:platform/default/distribute.bzl", "distribute_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "distribute",
    srcs = [
        "distributed_training_utils.py",
        "training_distributed.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:callbacks_v1",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:engine_utils",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:mode_keys",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras:saving",
        "//tensorflow/python/keras/mixed_precision/experimental:autocast_variable",
        "//tensorflow/python/keras/mixed_precision/experimental:policy",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "distribute_strategy_test_lib",
    srcs = [
        "distribute_strategy_test.py",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "distribute_strategy_test",
    srcs = ["distribute_strategy_test.py"],
    full_precision = True,
    main = "distribute_strategy_test.py",
    shard_count = 32,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":distribute_strategy_test_lib",
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
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:test",
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
        "no_gpu",  # b/129793413
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
        "no_gpu",  # b/129793413
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
        "no_gpu",  # b/129793413
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
        "no_gpu",  # b/129793413
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
        "no_pip",
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    full_precision = True,
    main = "keras_utils_test.py",
    shard_count = 32,
    tags = [
        "multi_and_single_gpu",
        "no_gpu",  # b/129793413
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_test_lib",
        "//tensorflow/python/keras/distribute:distribute_strategy_test_lib",
    ],
)

py_library(
    name = "keras_test_lib",
    srcs = [
        "keras_utils_test.py",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
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

distribute_py_test(
    name = "keras_two_input_layers_correctness_test",
    size = "medium",
    srcs = ["keras_two_input_layers_correctness_test.py"],
    full_precision = True,
    main = "keras_two_input_layers_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 5,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/117919883): Fix python error.
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
    ],
)

cuda_py_test(
    name = "multi_worker_test",
    srcs = ["multi_worker_test.py"],
    additional_deps = [":multi_worker_test_main_lib"],
    shard_count = 32,
    tags = [
        # TODO(b/124344198): Add "multi_and_single_gpu",
    ],
)

py_library(
    name = "multi_worker_test_main_lib",
    testonly = True,
    srcs = ["multi_worker_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
    ],
)
