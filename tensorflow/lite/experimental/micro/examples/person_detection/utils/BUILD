# Description:
#   TensorFlow Lite for Microcontrollers Vision Example Utils.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "raw_to_bitmap",
    srcs = ["raw_to_bitmap.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":raw_to_bitmap_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "raw_to_bitmap_lib",
    srcs = ["raw_to_bitmap.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "raw_to_bitmap_test",
    srcs = ["raw_to_bitmap_test.py"],
    data = glob(["testdata/**"]),
    python_version = "PY3",
    deps = [":raw_to_bitmap_lib"],
)
