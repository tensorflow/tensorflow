package(default_visibility = [
    "//tensorflow:internal",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "timeseries",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/timeseries/python/timeseries:py_init",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "timeseries_pip",
    deps = [
        ":timeseries",
        "//tensorflow/contrib/timeseries/examples:known_anomaly",
        "//tensorflow/contrib/timeseries/examples:lstm",
        "//tensorflow/contrib/timeseries/examples:multivariate",
        "//tensorflow/contrib/timeseries/examples:predict",
        "//tensorflow/contrib/timeseries/python/timeseries:test_utils",
        "//tensorflow/contrib/timeseries/python/timeseries/state_space_models:test_utils",
    ],
)
