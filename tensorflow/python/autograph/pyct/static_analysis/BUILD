load("//tensorflow:tensorflow.bzl", "py_test")

package(
    licenses = ["notice"],
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

py_library(
    name = "static_analysis",
    srcs = [
        "activity.py",
        "annos.py",
        "liveness.py",
        "reaching_definitions.py",
        "reaching_fndefs.py",
        "type_inference.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/utils",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "activity_test",
    srcs = ["activity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":static_analysis",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_library(
    name = "activity_test_lib",
    testonly = True,
    srcs = ["activity_test.py"],
    srcs_version = "PY3",
    deps = [
        ":static_analysis",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "activity_py3_test",
    srcs = ["activity_py3_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_oss_py35",
        # TODO(mdan): Remove this tag once no_oss_py35 is properly supported.
        "no_oss",
        "no_pip",
        "nopip",
    ],
    deps = [
        ":activity_test_lib",
        ":static_analysis",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "liveness_test",
    testonly = True,
    srcs = ["liveness_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":static_analysis",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_test(
    name = "reaching_definitions_test",
    srcs = ["reaching_definitions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":static_analysis",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_test(
    name = "type_inference_test",
    srcs = ["type_inference_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":static_analysis",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)
