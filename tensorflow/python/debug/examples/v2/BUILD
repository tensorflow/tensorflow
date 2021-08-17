package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_binary(
    name = "debug_fibonacci_v2",
    srcs = ["debug_fibonacci_v2.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_mnist_v2",
    srcs = ["debug_mnist_v2.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
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
        "v2only",
    ],
)
