load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

distribute_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
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
