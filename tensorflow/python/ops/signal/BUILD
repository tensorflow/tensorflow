load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "dct_ops",
    srcs = ["dct_ops.py"],
    strict_deps = True,
    deps = [
        ":fft_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:smart_cond",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    strict_deps = True,
)

py_library(
    name = "fft_ops",
    srcs = ["fft_ops.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:manip_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:spectral_ops_gen",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "mel_ops",
    srcs = ["mel_ops.py"],
    strict_deps = True,
    deps = [
        ":shape_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "mfcc_ops",
    srcs = ["mfcc_ops.py"],
    strict_deps = True,
    deps = [
        ":dct_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "reconstruction_ops",
    srcs = ["reconstruction_ops.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "shape_ops",
    srcs = ["shape_ops.py"],
    strict_deps = True,
    deps = [
        ":util_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "signal",
    srcs = ["signal.py"],
    strict_deps = True,
    deps = [
        ":__init__",
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

py_library(
    name = "spectral_ops",
    srcs = ["spectral_ops.py"],
    strict_deps = True,
    deps = [
        ":dct_ops",
        ":fft_ops",
        ":reconstruction_ops",
        ":shape_ops",
        ":window_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "util_ops",
    srcs = ["util_ops.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
    ],
)

py_library(
    name = "window_ops",
    srcs = ["window_ops.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)
