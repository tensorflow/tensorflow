load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")
load("//tensorflow/compiler/mlir/tfr:build_defs.bzl", "gen_op_libraries")
load("//tensorflow/core/platform:distribute.bzl", "distribute_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/tfr/...",
    ],
)

gen_op_libraries(
    name = "mnist_ops",
    src = "ops_defs.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/platform:flags",
        "@absl_py//absl:app",
    ],
)

tf_py_strict_test(
    name = "mnist_ops_test",
    size = "small",
    srcs = ["mnist_ops_test.py"],
    data = [":mnist_ops_mlir"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/170752141)
        "nomac",  # TODO(b/170752141)
    ],
    deps = [
        ":mnist_ops",
        ":mnist_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/tfr:test_utils",
        "//tensorflow/python/framework:load_library",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "mnist_train",
    srcs = ["mnist_train.py"],
    data = [":mnist_ops_mlir"],
    srcs_version = "PY3",
    deps = [
        ":mnist_ops",
        ":mnist_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:load_library",
        "@absl_py//absl/flags",
    ],
)

distribute_py_strict_test(
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
        "notpu",  # Takes too long (b/192305423)
        "notsan",  # Not needed, and there were issues with timeouts.
        "requires-net:external",
    ],

    # TODO(b/175056184): Re-enable xla_enable_strict_auto_jit once the issues
    # with GPU and the MLIR bridge are worked out.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":mnist_train",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/framework:test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)
