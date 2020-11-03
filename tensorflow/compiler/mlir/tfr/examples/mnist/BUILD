load("//tensorflow:tensorflow.bzl", "py_binary")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/compiler/mlir/tfr:build_defs.bzl", "gen_op_libraries")

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

py_binary(
    name = "mnist_train",
    srcs = ["mnist_train.py"],
    data = [":mnist_ops_mlir"],
    python_version = "PY3",
    deps = [
        ":mnist_ops",
        ":mnist_ops_py",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
