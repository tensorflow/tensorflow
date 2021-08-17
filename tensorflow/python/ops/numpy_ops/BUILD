load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "numpy",
    srcs = [
        "__init__.py",
        "np_array_ops.py",
        "np_arrays.py",
        "np_config.py",
        "np_dtypes.py",
        "np_export.py",
        "np_math_ops.py",
        "np_random.py",
        "np_utils.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sort_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "np_dtypes_test",
    srcs = ["np_dtypes_test.py"],
    deps = [
        ":numpy",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "np_arrays_test",
    srcs = ["np_arrays_test.py"],
    deps = [
        ":numpy",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "np_array_ops_test",
    srcs = ["np_array_ops_test.py"],
    deps = [
        ":numpy",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "np_logic_test",
    srcs = ["np_logic_test.py"],
    deps = [
        ":numpy",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "np_math_ops_test",
    srcs = ["np_math_ops_test.py"],
    deps = [
        ":numpy",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "np_random_test",
    srcs = ["np_random_test.py"],
    deps = [
        ":numpy",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "np_utils_test",
    srcs = ["np_utils_test.py"],
    deps = [
        ":numpy",
        "//tensorflow/python:platform",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "np_interop_test",
    srcs = ["np_interop_test.py"],
    deps = [
        ":numpy",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
