load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "mlir",
    srcs = ["mlir.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python:pywrap_mlir",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "mlir_test",
    srcs = ["mlir_test.py"],
    data = [
        "multi_add.tflite",
    ],
    strict_deps = True,
    tags = [
        "no_mac",  # TODO(b/291995580): Reenable.
        "no_pip",
    ],
    deps = [
        ":mlir",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python:pywrap_mlir",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
    ],
)
