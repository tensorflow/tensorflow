licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "lms_examples",
    srcs = ["cnn_mnist_lms.py",
            "mnist_deep_lms.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/lms:lms_py",
        "@six_archive//:six",
    ],
)
