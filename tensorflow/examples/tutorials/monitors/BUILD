# Example Estimator model

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "iris_monitors",
    srcs = [
        "iris_monitors.py",
    ],
    data = [
        "iris_test.csv",
        "iris_training.csv",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
