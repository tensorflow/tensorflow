package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "service",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:data_service_ops",
    ],
)
