load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_binary(
    name = "modify_model_interface",
    srcs = ["modify_model_interface.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":modify_model_interface_constants",
        ":modify_model_interface_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "modify_model_interface_lib",
    srcs = ["modify_model_interface_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_modify_model_interface",
        ":modify_model_interface_constants",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:schema_py",
    ],
)

py_test(
    name = "modify_model_interface_lib_test",
    srcs = ["modify_model_interface_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":modify_model_interface_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "modify_model_interface_constants",
    srcs = ["modify_model_interface_constants.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python:dtypes"],
)

pybind_extension(
    name = "_pywrap_modify_model_interface",
    srcs = ["modify_model_interface.cc"],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:modify_model_interface",
        "@pybind11",
    ],
)
