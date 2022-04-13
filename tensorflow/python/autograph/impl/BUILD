load("//tensorflow:tensorflow.bzl", "tf_py_test")

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
        "//tensorflow/python/autograph/converters",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/autograph/utils",
        "@gast_archive//:gast",
    ],
)

tf_py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "notsan",  # TODO(b/186153876)
    ],
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/utils",
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
