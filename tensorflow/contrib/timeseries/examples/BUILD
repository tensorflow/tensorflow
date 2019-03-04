load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "py_binary")

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
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [":predict_main_lib"],
)

py_library(
    name = "predict_main_lib",
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
        ":predict_main_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_binary(
    name = "known_anomaly",
    srcs = ["known_anomaly.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [":known_anomaly_main_lib"],
)

py_library(
    name = "known_anomaly_main_lib",
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
        ":known_anomaly_main_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_binary(
    name = "multivariate",
    srcs = ["multivariate.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [":multivariate_main_lib"],
)

py_library(
    name = "multivariate_main_lib",
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
        ":multivariate_main_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_binary(
    name = "lstm",
    srcs = ["lstm.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    visibility = ["//visibility:public"],
    deps = [":lstm_main_lib"],
)

py_library(
    name = "lstm_main_lib",
    srcs = ["lstm.py"],
    data = ["data/multivariate_periods.csv"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    visibility = ["//visibility:public"],
    deps = select({
        ":empty_condition": [],
        "//conditions:default": [],
    }) + [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/timeseries/python/timeseries:estimators",
        "//tensorflow/contrib/timeseries/python/timeseries:model",
        "//tensorflow/contrib/timeseries/python/timeseries:state_management",
    ],
)

py_test(
    name = "lstm_test",
    timeout = "long",  # Moderate but for asan
    srcs = ["lstm_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":lstm_main_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/estimator:estimator_py",
    ],
)
