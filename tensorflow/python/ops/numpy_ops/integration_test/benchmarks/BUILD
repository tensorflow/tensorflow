load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_binary(
    name = "micro_benchmarks",
    srcs = ["micro_benchmarks.py"],
    strict_deps = True,
    deps = [
        ":numpy_mlp",
        ":tf_numpy_mlp",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python:extra_py_tests_deps",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "numpy_mlp",
    srcs = ["numpy_mlp.py"],
    strict_deps = True,
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tf_numpy_mlp",
    srcs = ["tf_numpy_mlp.py"],
    strict_deps = True,
    deps = ["//tensorflow:tensorflow_py_no_contrib"],
)
