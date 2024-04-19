load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "gradient_checker_test",
    size = "medium",
    srcs = ["gradient_checker_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)
