load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    visibility = ["//visibility:private"],
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
        "//tensorflow/python/autograph/pyct",
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
