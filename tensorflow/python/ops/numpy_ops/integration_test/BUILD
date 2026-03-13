load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

py_test(
    name = "public_symbol_test",
    srcs = ["public_symbol_test.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "np_config_test",
    srcs = ["np_config_test.py"],
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/ops/numpy_ops:np_config",
    ],
)
