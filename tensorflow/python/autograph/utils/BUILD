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
    name = "type_registry",
    srcs = [
        "type_registry.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "utils",
    srcs = [
        "__init__.py",
        "ag_logging.py",
        "context_managers.py",
        "misc.py",
        "tensor_list.py",
        "tensors.py",
        "testing.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_test(
    name = "context_managers_test",
    srcs = ["context_managers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "misc_test",
    srcs = ["misc_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "tensor_list_test",
    srcs = ["tensor_list_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
    ],
)

py_test(
    name = "tensors_test",
    srcs = ["tensors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)
