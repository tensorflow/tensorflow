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
    name = "core",
    srcs = [
        "ag_ctx.py",
        "config.py",
        "config_lib.py",
        "converter.py",
        "function_wrappers.py",
        "unsupported_features_checker.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/autograph/utils:ag_logging",
        "@gast_archive//:gast",
    ],
)

py_library(
    name = "test_lib",
    srcs = [
        "converter_testing.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":core",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":core",
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "function_wrappers_test",
    srcs = ["function_wrappers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":core",
        "//tensorflow/python:client_testlib",
    ],
)
