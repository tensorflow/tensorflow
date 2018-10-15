licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "ipu",
    srcs = [
      "__init__.py",
      "python/ops.py",
      "python/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops",
        "//tensorflow/contrib/compiler:compiler_py",
    ],
)

tf_py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/ops_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "while_loop_perf_test",
    size = "small",
    srcs = ["python/while_loop_perf_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/compiler/plugin/poplar:ipu_ops",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "mapping_test",
    size = "small",
    srcs = ["python/mapping_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/compiler/plugin/poplar:ipu_ops",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "conditional_test",
    size = "small",
    srcs = ["python/conditional_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "resnet18_nobn_test",
    size = "medium",
    srcs = ["size_speed_tests/resnet18_nobn_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "while_loop_test",
    size = "medium",
    srcs = ["python/while_loop_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "ops_test",
        "while_loop_perf_test",
        "conditional_test",
        "mapping_test",
        "while_loop_test",
    ],
)

test_suite(
    name = "size_speed_tests",
    tests = [
        "resnet18_nobn_test",
    ],
)

test_suite(
    name = "training_tests",
    tests = [
    ],
)

test_suite(
    name = "all_ci_tests",
    tests = [
        ":all_tests",
        ":size_speed_tests",
        ":training_tests",
    ],
)
