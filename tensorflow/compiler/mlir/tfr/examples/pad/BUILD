load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")
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
    name = "pad_ops",
    src = "ops_defs.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/platform:flags",
        "@absl_py//absl:app",
    ],
)

tf_py_strict_test(
    name = "pad_ops_test",
    size = "medium",
    srcs = ["pad_ops_test.py"],
    data = [":pad_ops_mlir"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/170752141)
        "nomac",  # TODO(b/170752141)
    ],
    deps = [
        ":pad_ops",
        ":pad_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/tfr:test_utils",
        "//tensorflow/python/framework:load_library",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
