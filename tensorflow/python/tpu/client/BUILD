# Cloud TPU Client.

load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "client",
    srcs = [
        "client.py",
        "version.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "client_lib",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [":client"],
)

tf_py_test(
    name = "client_py_test",
    size = "small",
    srcs = ["client_test.py"],
    grpc_enabled = True,
    main = "client_test.py",
    python_version = "PY3",
    deps = [
        ":client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training_server_lib",
    ],
)
