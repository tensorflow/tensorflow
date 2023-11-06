load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//third_party/py/tensorflow_numerics:__subpackages__",
        "//third_party/py/trax/tf_numpy/numpy:__pkg__",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "numpy",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":np_array_ops",
        ":np_arrays",
        ":np_config",
        ":np_dtypes",
        ":np_math_ops",
        ":np_random",
        ":np_utils",
        "//tensorflow/python/ops:array_ops",
    ],
)

py_strict_library(
    name = "np_utils",
    srcs = ["np_utils.py"],
    tags = [
        "ignore_for_dep=third_party.py.requests",
    ],
    deps = [
        ":np_arrays",
        ":np_dtypes",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_array_ops",
    srcs = ["np_array_ops.py"],
    deps = [
        ":np_arrays",
        ":np_dtypes",
        ":np_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:manip_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sort_ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_config",
    srcs = ["np_config.py"],
    deps = [
        ":np_dtypes",
        ":np_math_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:weak_tensor_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "np_dtypes",
    srcs = ["np_dtypes.py"],
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_random",
    srcs = ["np_random.py"],
    deps = [
        ":np_array_ops",
        ":np_dtypes",
        ":np_utils",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/util:tf_export",
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
        ":np_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bitwise_ops",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:sort_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "np_arrays",
    srcs = ["np_arrays.py"],
    deps = [
        ":np_dtypes",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
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
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
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
        "//tensorflow/python/framework:tensor",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
