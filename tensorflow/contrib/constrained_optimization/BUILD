package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "constrained_optimization_pip",
    deps = [
        ":constrained_optimization",
        ":test_util",
    ],
)

py_library(
    name = "constrained_optimization",
    srcs = [
        "__init__.py",
        "python/candidates.py",
        "python/constrained_minimization_problem.py",
        "python/constrained_optimizer.py",
        "python/external_regret_optimizer.py",
        "python/swap_regret_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "candidates_test",
    srcs = ["python/candidates_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        # TODO(b/129496144): Re-enable MSAN test.
        "nomsan",
    ],
    deps = [
        ":constrained_optimization",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

# NOTE: This library can't be "testonly" since it needs to be included in the
# pip package.
py_library(
    name = "test_util",
    srcs = ["python/test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constrained_optimization",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:standard_ops",
    ],
)

py_test(
    name = "external_regret_optimizer_test",
    srcs = ["python/external_regret_optimizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constrained_optimization",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "swap_regret_optimizer_test",
    srcs = ["python/swap_regret_optimizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constrained_optimization",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)
