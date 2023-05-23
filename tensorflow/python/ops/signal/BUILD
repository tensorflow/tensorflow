load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "dct_ops",
    srcs = ["dct_ops.py"],
    deps = [
        ":fft_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:smart_cond",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "fft_ops",
    srcs = ["fft_ops.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:spectral_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "mel_ops",
    srcs = ["mel_ops.py"],
    deps = [
        ":shape_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "mfcc_ops",
    srcs = ["mfcc_ops.py"],
    deps = [
        ":dct_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "reconstruction_ops",
    srcs = ["reconstruction_ops.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "shape_ops",
    srcs = ["shape_ops.py"],
    deps = [
        ":util_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "signal",
    srcs = ["signal.py"],
    deps = [
        ":dct_ops",
        ":fft_ops",
        ":mel_ops",
        ":mfcc_ops",
        ":reconstruction_ops",
        ":shape_ops",
        ":spectral_ops",
        ":window_ops",
    ],
)

py_strict_library(
    name = "spectral_ops",
    srcs = ["spectral_ops.py"],
    deps = [
        ":dct_ops",
        ":fft_ops",
        ":reconstruction_ops",
        ":shape_ops",
        ":window_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "util_ops",
    srcs = ["util_ops.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "window_ops",
    srcs = ["window_ops.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)
