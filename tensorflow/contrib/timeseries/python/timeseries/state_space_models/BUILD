# State space components and ensembles

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "state_space_model",
    srcs = ["state_space_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kalman_filter",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/timeseries/python/timeseries:feature_keys",
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/contrib/timeseries/python/timeseries:model",
        "//tensorflow/contrib/timeseries/python/timeseries:model_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator:estimator_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "state_space_model_test",
    timeout = "long",  # Moderate but for asan
    srcs = ["state_space_model_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":state_space_model",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/timeseries/python/timeseries:estimators",
        "//tensorflow/contrib/timeseries/python/timeseries:feature_keys",
        "//tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/contrib/timeseries/python/timeseries:saved_model_utils",
        "//tensorflow/contrib/timeseries/python/timeseries:state_management",
        "//tensorflow/contrib/timeseries/python/timeseries:test_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "kalman_filter",
    srcs = ["kalman_filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:numerics",
    ],
)

tf_py_test(
    name = "kalman_filter_test",
    srcs = ["kalman_filter_test.py"],
    additional_deps = [
        ":kalman_filter",
        "//third_party/py/numpy",
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "level_trend",
    srcs = ["level_trend.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":state_space_model",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
    ],
)

tf_py_test(
    name = "level_trend_test",
    srcs = ["level_trend_test.py"],
    additional_deps = [
        ":level_trend",
        ":test_utils",
        ":state_space_model",
        "//tensorflow/contrib/timeseries/python/timeseries:test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "periodic",
    srcs = ["periodic.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":state_space_model",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "periodic_test",
    srcs = ["periodic_test.py"],
    additional_deps = [
        ":periodic",
        ":test_utils",
        ":state_space_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "structural_ensemble",
    srcs = ["structural_ensemble.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":level_trend",
        ":periodic",
        ":state_space_model",
        ":varma",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "structural_ensemble_test",
    timeout = "long",  # Moderate but for asan/tsan timeouts
    srcs = ["structural_ensemble_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":state_space_model",
        ":structural_ensemble",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/timeseries/python/timeseries:estimators",
        "//tensorflow/contrib/timeseries/python/timeseries:feature_keys",
        "//tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/estimator:estimator_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "varma",
    srcs = ["varma.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":state_space_model",
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
    ],
)

tf_py_test(
    name = "varma_test",
    srcs = ["varma_test.py"],
    additional_deps = [
        ":state_space_model",
        ":varma",
        "//tensorflow/contrib/timeseries/python/timeseries:feature_keys",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "filtering_postprocessor",
    srcs = ["filtering_postprocessor.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "filtering_postprocessor_test",
    srcs = ["filtering_postprocessor_test.py"],
    additional_deps = [
        ":filtering_postprocessor",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/timeseries/python/timeseries:math_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
