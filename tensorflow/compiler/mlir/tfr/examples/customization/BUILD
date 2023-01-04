load("//tensorflow:tensorflow.default.bzl", "tf_py_test")
load("//tensorflow/compiler/mlir/tfr:build_defs.bzl", "gen_op_libraries")

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
    name = "test_ops",
    src = "ops_defs.py",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "test_ops_test",
    size = "small",
    srcs = ["test_ops_test.py"],
    data = [":test_ops_mlir"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/170752141)
        "nomac",  # TODO(b/170752141)
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/tfr:test_utils",
        "//tensorflow/python:test_ops",
    ],
)
