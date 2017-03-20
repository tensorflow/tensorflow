package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "estimator_py",
    srcs = ["estimator_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":checkpoint_utils",
        ":estimator",
        ":export",
        ":inputs",
        ":model_fn",
        ":run_config",
        "//tensorflow/python:util",
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

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["checkpoint_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":checkpoint_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "model_fn",
    srcs = ["model_fn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
        "@six_archive//:six",
    ],
)

py_test(
    name = "model_fn_test",
    size = "small",
    srcs = ["model_fn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_fn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "run_config",
    srcs = ["run_config.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":checkpoint_utils",
        ":export",
        ":model_fn",
        ":run_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:metrics",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":model_fn",
        ":numpy_io",
        ":run_config",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:layers",
        "//tensorflow/python:saver_test_utils",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:loader",
    ],
)

py_library(
    name = "export_output",
    srcs = ["export/export_output.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/saved_model:signature_def_utils",
    ],
)

py_test(
    name = "export_output_test",
    size = "small",
    srcs = ["export/export_output_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_library(
    name = "export",
    srcs = [
        "export/export_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        ":export_output",
    ],
)

py_library(
    name = "export_export",
    srcs = [
        "export/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_test(
    name = "export_test",
    size = "small",
    srcs = ["export/export_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        ":export_output",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs/inputs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_io",
        ":pandas_io",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["inputs/numpy_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
    ],
)

py_test(
    name = "numpy_io_test",
    size = "small",
    srcs = ["inputs/numpy_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_io",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["inputs/pandas_io.py"],
    srcs_version = "PY2AND3",
    deps = [":inputs_queues"],
)

py_test(
    name = "pandas_io_test",
    size = "small",
    srcs = ["inputs/pandas_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pandas_io",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "inputs_queues",
    srcs = [
        "inputs/queues/__init__.py",
        "inputs/queues/feeding_functions.py",
        "inputs/queues/feeding_queue_runner.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:training"],
)

py_test(
    name = "feeding_functions_test",
    size = "small",
    srcs = [
        "inputs/queues/feeding_functions_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "feeding_queue_runner_test",
    size = "small",
    srcs = ["inputs/queues/feeding_queue_runner_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
    ],
)
