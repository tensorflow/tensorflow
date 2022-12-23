load("//tensorflow:pytype.default.bzl", "pytype_library")
load(
    "//tensorflow/dtensor:build_defs.bzl",
    "TPU_V3_DONUT_BACKEND",
    "dtensor_test",
)

# File used by internal tests.
exports_files(["spmd_test.py"])

pytype_library(
    name = "test_util",
    testonly = False,  # build_pip_package depends on this target.
    srcs = [
        "test_backend_name.py",
        "test_util.py",
        "test_util_ops.py",
    ],
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/dtensor:dtensor-users",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:bitwise_ops_gen",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:spectral_ops_gen",
        "//tensorflow/python:stateless_random_ops_gen",
        "//tensorflow/python:stateless_random_ops_v2_gen",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "config_test",
    srcs = ["config_test.py"],
    main = "config_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)

dtensor_test(
    name = "numpy_util_test",
    srcs = ["numpy_util_test.py"],
    main = "numpy_util_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

dtensor_test(
    name = "spmd_test",
    srcs = ["spmd_test.py"],
    additional_backends = [TPU_V3_DONUT_BACKEND],
    main = "spmd_test.py",
    shard_count = {
        "cpu": 25,
        "gpu": 10,
        "tpu": 10,
        TPU_V3_DONUT_BACKEND: 32,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:bitwise_ops_gen",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:linalg_ops_gen",
        "//tensorflow/python:list_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:stateless_random_ops_gen",
        "//tensorflow/python:string_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
