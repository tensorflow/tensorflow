load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

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
    name = "impl",
    srcs = [
        "api.py",
        "conversion.py",
    ],
    srcs_version = "PY3",
    # TODO(mdan): fix the cyclic dependency issue when build_cleaner is applied on this library.
    tags = ["nofixdeps"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/converters:asserts",
        "//tensorflow/python/autograph/converters:break_statements",
        "//tensorflow/python/autograph/converters:call_trees",
        "//tensorflow/python/autograph/converters:conditional_expressions",
        "//tensorflow/python/autograph/converters:continue_statements",
        "//tensorflow/python/autograph/converters:control_flow",
        "//tensorflow/python/autograph/converters:directives",
        "//tensorflow/python/autograph/converters:functions",
        "//tensorflow/python/autograph/converters:lists",
        "//tensorflow/python/autograph/converters:logical_expressions",
        "//tensorflow/python/autograph/converters:return_statements",
        "//tensorflow/python/autograph/converters:slices",
        "//tensorflow/python/autograph/converters:variables",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/autograph/utils:ag_logging",
        "@gast_archive//:gast",
    ],
)

tf_py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "conversion_test",
    srcs = ["conversion_test.py"],
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/impl/testing:pybind_for_testing",
        "@gast_archive//:gast",
    ],
)
