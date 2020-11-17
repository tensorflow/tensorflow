load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/compiler/mlir/tfr:build_defs.bzl", "gen_op_libraries")
load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/mlir/tfr/...",
    ],
)

gen_op_libraries(
    name = "mnist_ops",
    src = "ops_defs.py",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "mnist_ops_test",
    size = "small",
    srcs = ["mnist_ops_test.py"],
    data = [":mnist_ops_mlir"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/170752141)
        "nomac",  # TODO(b/170752141)
    ],
    deps = [
        ":mnist_ops",
        ":mnist_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/tfr:test_utils",
    ],
)

py_library(
    name = "mnist_train",
    srcs = ["mnist_train.py"],
    data = [":mnist_ops_mlir"],
    deps = [
        ":mnist_ops",
        ":mnist_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "@absl_py//absl/flags",
    ],
)

distribute_py_test(
    name = "mnist_train_test",
    size = "medium",
    srcs = ["mnist_train_test.py"],
    data = [":mnist_ops_mlir"],
    disable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # b/173431253
        "no_oss",
        "notsan",  # b/173246447
    ],
    xla_enable_strict_auto_jit = False,  # b/173254861
    deps = [
        ":mnist_train",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)
