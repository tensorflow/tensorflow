licenses(["notice"])  # 3-clause BSD.

py_library(
    name = "spinn",
    srcs = ["spinn.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow/contrib/eager/python/examples/spinn:data",
        "@six_archive//:six",
    ],
)
