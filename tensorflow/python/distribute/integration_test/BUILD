load("//tensorflow/core/platform:distribute.bzl", "distribute_py_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

distribute_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    tags = [
        "no_windows",  # TODO(b/171350360)
        "nomultivm",  # multi_worker_test_base incompatible with multivm base
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "mwms_peer_failure_test",
    size = "medium",
    srcs = ["mwms_peer_failure_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/227372713)
        "notsan",  # b/195248428
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "mwms_peer_failure_test_lib",
    srcs = ["mwms_peer_failure_test.py"],
    visibility = ["//learning/brain/runtime/python:__pkg__"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/eager:test",
    ],
)

tpu_py_test(
    name = "tpu_memory_test",
    size = "medium",
    srcs = ["tpu_memory_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = True,
    disable_tfrt = False,
    disable_v2 = True,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)
