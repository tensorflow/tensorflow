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
    name = "examples_v1_test",
    size = "medium",
    srcs = ["examples_v1_test.sh"],
    data = [
        ":debug_errors",
        ":debug_fibonacci",
        ":debug_keras",
        ":debug_mnist",
        ":debug_tflearn_iris",
        "//tensorflow/python/debug/cli:offline_analyzer",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/143150907)
        "nomsan",  # TODO(b/143150907)
        "v1only",
    ],
)
