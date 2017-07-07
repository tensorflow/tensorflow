load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "predict",
    srcs = [
        "predict.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "predict_test",
    srcs = [
        "predict_test.py",
    ],
    data = [
        "data/period_trend.csv",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip_gpu",  # b/63391119
    ],
    deps = [
        ":predict",
    ],
)

py_binary(
    name = "known_anomaly",
    srcs = [
        "known_anomaly.py",
    ],
    data = [
        "data/changepoints.csv",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "known_anomaly_test",
    srcs = [
        "known_anomaly_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip_gpu",  # b/63391119
    ],
    deps = [
        ":known_anomaly",
    ],
)

py_binary(
    name = "multivariate",
    srcs = [
        "multivariate.py",
    ],
    data = [
        "data/multivariate_level.csv",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "multivariate_test",
    srcs = [
        "multivariate_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip_gpu",  # b/63391119
    ],
    deps = [
        ":multivariate",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
