package(default_visibility = [
    "//learning/brain/contrib/kfac/examples:__subpackages__",
    "//tensorflow/contrib/kfac/examples:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "mlp_mnist_main",
    srcs = ["mlp_mnist_main.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mlp",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mnist",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "convnet_mnist_single_main",
    srcs = ["convnet_mnist_single_main.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "convnet_mnist_multi_tower_main",
    srcs = ["convnet_mnist_multi_tower_main.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "convnet_mnist_distributed_main",
    srcs = ["convnet_mnist_distributed_main.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "convnet",
    srcs = ["convnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mlp",
        ":mnist",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
