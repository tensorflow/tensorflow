package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

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
    name = "estimator_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dnn",
        ":dnn_linear_combined",
        ":extenders",
        ":head",
        ":linear",
        ":logit_fns",
        ":multi_head",
        ":replicate_model_fn",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "dnn",
    srcs = ["python/estimator/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:nn",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn",
    ],
)

py_test(
    name = "dnn_test",
    size = "small",
    srcs = ["python/estimator/dnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dnn",
        ":head",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:dnn_testing_utils",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["python/estimator/dnn_linear_combined.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:nn",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn_linear_combined",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "medium",
    srcs = ["python/estimator/dnn_linear_combined_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dnn_linear_combined",
        ":head",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:dnn_testing_utils",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:linear_testing_utils",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "extenders",
    srcs = [
        "python/estimator/extenders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:util",
        "@six_archive//:six",
    ],
)

py_test(
    name = "extenders_test",
    size = "medium",
    srcs = ["python/estimator/extenders_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/62863147
    deps = [
        ":extenders",
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:linear",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_test(
    name = "head_test",
    size = "small",
    srcs = ["python/estimator/head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:signature_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "linear",
    srcs = ["python/estimator/linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:linear",
    ],
)

py_test(
    name = "linear_test",
    size = "small",
    srcs = ["python/estimator/linear_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":head",
        ":linear",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:linear_testing_utils",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "logit_fns",
    srcs = [
        "python/estimator/logit_fns.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/estimator:dnn",
        "//tensorflow/python/estimator:linear",
        "//tensorflow/python/estimator:util",
    ],
)

py_test(
    name = "logit_fns_test",
    size = "small",
    srcs = ["python/estimator/logit_fns_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logit_fns",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:session",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_library(
    name = "multi_head",
    srcs = [
        "python/estimator/multi_head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/saved_model:signature_constants",
        "@six_archive//:six",
    ],
)

py_test(
    name = "multi_head_test",
    size = "small",
    srcs = ["python/estimator/multi_head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        ":multi_head",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/saved_model:signature_constants",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "replicate_model_fn",
    srcs = [
        "python/estimator/replicate_model_fn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:device_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:util",
        "//tensorflow/python/ops/losses",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "replicate_model_fn_test",
    size = "medium",
    srcs = ["python/estimator/replicate_model_fn_test.py"],
    additional_deps = [
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:optimizers",
        "//tensorflow/python/estimator:prediction_keys",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        ":replicate_model_fn",
    ],
    tags = [
        "manual",
        "multi_gpu",
        "notap",
    ],
)
