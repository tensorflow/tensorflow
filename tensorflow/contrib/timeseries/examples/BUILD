load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

config_setting(
    name = "empty_condition",
    values = {"define": "UNUSED=unused"},
)

py_binary(
    name = "predict",
    srcs = ["predict.py"],
    data = ["data/period_trend.csv"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = select({
        ":empty_condition": [],
        "//conditions:default": [],
    }) + [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "predict_test",
    timeout = "long",  # Moderate but for asan
    srcs = ["predict_test.py"],
    data = ["data/period_trend.csv"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",  # TODO: needs investigation on Windows
        "notsan",  # b/67513579
    ],
    deps = [
        ":predict",
        "//tensorflow/python:client_testlib",
    ],
)

py_binary(
    name = "known_anomaly",
    srcs = ["known_anomaly.py"],
    data = ["data/changepoints.csv"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = select({
        ":empty_condition": [],
        "//conditions:default": [],
    }) + [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "known_anomaly_test",
    timeout = "long",  # Moderate but for asan
    srcs = ["known_anomaly_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":known_anomaly",
        "//tensorflow/python:client_testlib",
    ],
)

py_binary(
    name = "multivariate",
    srcs = ["multivariate.py"],
    data = ["data/multivariate_level.csv"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = select({
        ":empty_condition": [],
        "//conditions:default": [],
    }) + [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "multivariate_test",
    timeout = "long",  # Moderate but for asan
    srcs = [
        "multivariate_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":multivariate",
        "//tensorflow/python:client_testlib",
    ],
)

py_binary(
    name = "lstm",
    srcs = ["lstm.py"],
    data = ["data/multivariate_periods.csv"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = select({
        ":empty_condition": [],
        "//conditions:default": [],
    }) + [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/timeseries/python/timeseries:estimators",
        "//tensorflow/contrib/timeseries/python/timeseries:model",
    ],
)

py_test(
    name = "lstm_test",
    timeout = "long",  # Moderate but for asan
    srcs = ["lstm_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":lstm",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/estimator:estimator_py",
    ],
)
