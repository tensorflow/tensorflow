load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

distribute_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    disable_mlir_bridge = False,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_combinations",
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
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/eager:test",
    ],
)
