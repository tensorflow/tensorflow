load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_test(
    name = "gradient_checker_test",
    size = "medium",
    srcs = ["gradient_checker_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)
