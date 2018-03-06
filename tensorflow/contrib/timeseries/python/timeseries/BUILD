load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "py_init",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ar_model",
        ":estimators",
        ":feature_keys",
        ":input_pipeline",
        ":saved_model_utils",
    ],
)

py_library(
    name = "feature_keys",
    srcs = [
        "feature_keys.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python/saved_model:signature_constants"],
)

py_library(
    name = "saved_model_utils",
    srcs = [
        "saved_model_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":head",
        ":input_pipeline",
        ":model_utils",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":math_utils",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "estimators",
    srcs = [
        "estimators.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ar_model",
        ":feature_keys",
        ":head",
        ":math_utils",
        ":state_management",
        "//tensorflow/contrib/timeseries/python/timeseries/state_space_models:filtering_postprocessor",
        "//tensorflow/contrib/timeseries/python/timeseries/state_space_models:state_space_model",
        "//tensorflow/contrib/timeseries/python/timeseries/state_space_models:structural_ensemble",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:export",
    ],
)

py_test(
    name = "estimators_test",
    timeout = "long",
    srcs = [
        "estimators_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip_gpu",  # b/63391119
        "nomsan",  # Takes too long to run.
        "notsan",  # b/67865658
    ],
    deps = [
        ":ar_model",
        ":estimators",
        ":feature_keys",
        ":input_pipeline",
        ":saved_model_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "head",
    srcs = [
        "head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:export",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
    ],
)

py_test(
    name = "head_test",
    srcs = [
        "head_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip_gpu"],  # b/63391119
    deps = [
        ":feature_keys",
        ":head",
        ":model",
        ":state_management",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "model_utils",
    srcs = [
        "model_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "model_utils_test",
    srcs = [
        "model_utils_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip_gpu",  # b/63391119
    ],
    deps = [
        ":model_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "state_management",
    srcs = [
        "state_management.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":math_utils",
        ":model",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_test(
    name = "state_management_test",
    srcs = [
        "state_management_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",  # b/64527635
        "no_pip_gpu",  # b/63391119
    ],
    deps = [
        ":feature_keys",
        ":input_pipeline",
        ":math_utils",
        ":model",
        ":state_management",
        ":test_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "input_pipeline",
    srcs = [
        "input_pipeline.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":model_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator:estimator_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "input_pipeline_test",
    srcs = [
        "input_pipeline_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # b/63709811
        "no_pip",  # b/63709811
        "no_pip_gpu",  # b/63391119
    ],
    deps = [
        ":feature_keys",
        ":input_pipeline",
        ":test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "test_utils",
    srcs = [
        "test_utils.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":estimators",
        ":feature_keys",
        ":input_pipeline",
        ":state_management",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "ar_model",
    srcs = [
        "ar_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":model",
        ":model_utils",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_test(
    name = "ar_model_test",
    timeout = "long",  # Moderate but for asan
    srcs = [
        "ar_model_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":ar_model",
        ":estimators",
        ":feature_keys",
        ":input_pipeline",
        ":test_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "math_utils",
    srcs = [
        "math_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "math_utils_test",
    srcs = [
        "math_utils_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip_gpu",  # b/63391119
        "no_windows",  # TODO: needs investigation on Windows
        "nomsan",
    ],
    deps = [
        ":feature_keys",
        ":input_pipeline",
        ":math_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
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
