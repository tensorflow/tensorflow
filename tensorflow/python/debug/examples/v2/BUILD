load("//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "debug_fibonacci_v2",
    srcs = ["debug_fibonacci_v2.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "debug_mnist_v2",
    srcs = ["debug_mnist_v2.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

sh_test(
    name = "examples_v2_test",
    size = "medium",
    srcs = ["examples_v2_test.sh"],
    data = [
        ":debug_fibonacci_v2",
        ":debug_mnist_v2",
        "//tensorflow/python/debug/cli:offline_analyzer",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/143150907)
        "nomsan",  # TODO(b/143150907)
        "requires-mem:16g",
        "v2only",
    ],
)
