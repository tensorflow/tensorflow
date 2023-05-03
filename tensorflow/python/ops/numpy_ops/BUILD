load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "numpy",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":np_array_ops",
        ":np_arrays",
        ":np_config",
        ":np_dtypes",
        ":np_math_ops",
        ":np_random",
        ":np_utils",
        "//tensorflow/python:array_ops",
    ],
)

py_strict_library(
    name = "np_utils",
    srcs = ["np_utils.py"],
    deps = [
        ":np_arrays",
        ":np_dtypes",
        ":np_export",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:indexed_slices",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_array_ops",
    srcs = ["np_array_ops.py"],
    deps = [
        ":np_arrays",
        ":np_dtypes",
        ":np_export",
        ":np_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sort_ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_config",
    srcs = ["np_config.py"],
    deps = [
        ":np_dtypes",
        ":np_export",
        ":np_math_ops",
        "//tensorflow/python:framework_ops",
    ],
)

py_strict_library(
    name = "np_dtypes",
    srcs = ["np_dtypes.py"],
    deps = [
        ":np_export",
        "//tensorflow/python:dtypes",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_export",
    srcs = ["np_export.py"],
    deps = ["//tensorflow/python/util:tf_export"],
)

py_strict_library(
    name = "np_random",
    srcs = ["np_random.py"],
    deps = [
        ":np_array_ops",
        ":np_dtypes",
        ":np_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/framework:random_seed",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_math_ops",
    srcs = ["np_math_ops.py"],
    deps = [
        ":np_array_ops",
        ":np_arrays",
        ":np_dtypes",
        ":np_export",
        ":np_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:sort_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:while_loop",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_arrays",
    srcs = ["np_arrays.py"],
    deps = [
        ":np_dtypes",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/framework:tensor_conversion",
    ],
)

cuda_py_strict_test(
    name = "np_dtypes_test",
    srcs = ["np_dtypes_test.py"],
    deps = [
        ":np_dtypes",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "np_arrays_test",
    srcs = ["np_arrays_test.py"],
    deps = [
        ":np_arrays",
        ":np_math_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "np_array_ops_test",
    srcs = ["np_array_ops_test.py"],
    tags = [
        "no_windows",  # TODO(b/215381493)
    ],
    deps = [
        ":np_array_ops",
        ":np_arrays",
        ":np_math_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "np_logic_test",
    srcs = ["np_logic_test.py"],
    deps = [
        ":np_array_ops",
        ":np_arrays",
        ":np_math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "np_math_ops_test",
    srcs = ["np_math_ops_test.py"],
    deps = [
        ":np_array_ops",
        ":np_arrays",
        ":np_math_ops",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "np_random_test",
    srcs = ["np_random_test.py"],
    deps = [
        ":np_array_ops",
        ":np_dtypes",
        ":np_math_ops",
        ":np_random",
        ":numpy",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "np_utils_test",
    srcs = ["np_utils_test.py"],
    deps = [
        ":np_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "np_interop_test",
    srcs = ["np_interop_test.py"],
    deps = [
        ":np_math_ops",
        ":numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
