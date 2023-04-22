package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_binary(
    name = "debug_fibonacci",
    srcs = ["debug_fibonacci.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_errors",
    srcs = ["debug_errors.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_tflearn_iris",
    srcs = ["debug_tflearn_iris.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_keras",
    srcs = ["debug_keras.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_mnist",
    srcs = ["debug_mnist_v1.py"],
    main = "debug_mnist_v1.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

sh_test(
    name = "examples_v1_debug_errors_test",
    srcs = ["examples_v1_debug_errors_test.sh"],
    data = [
        ":debug_errors",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/190625515)
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_fibonacci_test",
    srcs = ["examples_v1_debug_fibonacci_test.sh"],
    data = [
        ":debug_fibonacci",
    ],
    tags = [
        "no_windows",
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_keras_test",
    size = "medium",
    srcs = ["examples_v1_debug_keras_test.sh"],
    data = [
        ":debug_keras",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/190625515)
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_mnist_test",
    srcs = ["examples_v1_debug_mnist_test.sh"],
    data = [
        ":debug_mnist",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/193153560)
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_tflearn_iris_test",
    srcs = ["examples_v1_debug_tflearn_iris_test.sh"],
    data = [
        ":debug_tflearn_iris",
    ],
    tags = [
        "no_windows",
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_offline_analyzer_test",
    srcs = ["examples_v1_offline_analyzer_test.sh"],
    data = [
        "//tensorflow/python/debug/cli:offline_analyzer",
    ],
    tags = [
        "no_windows",
        "v1only",
    ],
)
