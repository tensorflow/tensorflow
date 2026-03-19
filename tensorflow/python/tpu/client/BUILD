# Cloud TPU Client.

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

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
    strict_deps = True,
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "client_lib",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [":client"],
)

tf_py_strict_test(
    name = "client_py_test",
    size = "small",
    srcs = ["client_test.py"],
    grpc_enabled = True,
    main = "client_test.py",
    deps = [
        ":client",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/flags",
    ],
)
