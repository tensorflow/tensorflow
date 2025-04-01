# Cloud TPU Client.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "client",
    srcs = [
        "client.py",
        "version.py",
    ],
    deps = ["@absl_py//absl/flags"],
)

py_strict_library(
    name = "client_lib",
    srcs = [
        "__init__.py",
    ],
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
