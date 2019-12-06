# Description:
#   TensorFlow Lite for Microcontrollers Vision Example Utils.
licenses(["notice"])  # Apache 2.0

py_binary(
    name = "raw_to_bitmap",
    srcs = ["raw_to_bitmap.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = ["//third_party/py/numpy"],
)

py_library(
    name = "raw_to_bitmap_lib",
    srcs = ["raw_to_bitmap.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "raw_to_bitmap_test",
    srcs = ["raw_to_bitmap_test.py"],
    data = glob(["testdata/**"]),
    python_version = "PY3",
    tags = ["noubsan"],  # TODO(b/144512025): Fix raw_to_bitmap_test to fix ubsan failure.
    deps = [
        ":raw_to_bitmap_lib",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)
