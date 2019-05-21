# Example Estimator model

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "mnist_tflite",
    srcs = [
        "dataset.py",
        "mnist_tflite.py",
    ],
    python_version = "PY2",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
