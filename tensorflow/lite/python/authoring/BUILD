package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python/metrics:converter_error_data_proto_py",
        "//tensorflow/python:framework",
    ],
)

py_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":authoring",
        "//tensorflow:tensorflow_py",
    ],
)
