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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    disable_v3 = True,  # Not needed. Save some resources and test time.
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # Not needed, and there were issues with timeouts.
        "no_oss",  # Avoid downloading mnist data set in oss.
        "nomultivm",  # Not needed. Save some resources and test time.
        "notap",  # The test is too long to run as part of llvm presubmits (b/173661843).
        "notsan",  # Not needed, and there were issues with timeouts.
    ],

    # TODO(b/175056184): Re-enable xla_enable_strict_auto_jit once the issues
    # with GPU and the MLIR bridge are worked out.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":mnist_train",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:is_mlir_bridge_test_true",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)
