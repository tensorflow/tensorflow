licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

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

py_library(
    name = "static_analysis",
    srcs = [
        "activity.py",
        "annos.py",
        "cfg.py",  # TODO(mdan): Remove.
        "live_values.py",
        "liveness.py",
        "reaching_definitions.py",
        "type_info.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/contrib/autograph/utils",
        "//tensorflow/python:util",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "activity_test",
    srcs = ["activity_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "cfg_test",
    srcs = ["cfg_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "live_values_test",
    srcs = ["live_values_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

# TODO(mdan): Enable these tests once child change is in.
py_test(
    name = "liveness_test",
    srcs = ["liveness_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "reaching_definitions_test",
    srcs = ["reaching_definitions_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "type_info_test",
    srcs = ["type_info_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/contrib/autograph/utils",
        "//tensorflow/python:client_testlib",
    ],
)
