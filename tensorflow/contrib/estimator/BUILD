package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "estimator_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":baseline",
        ":boosted_trees",
        ":dnn",
        ":dnn_linear_combined",
        ":early_stopping",
        ":export",
        ":exporter",
        ":extenders",
        ":head",
        ":hooks",
        ":linear",
        ":logit_fns",
        ":multi_head",
        ":replicate_model_fn",
        ":rnn",
        ":saved_model_estimator",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "baseline",
    srcs = ["python/estimator/baseline.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:baseline",
    ],
)

py_test(
    name = "baseline_test",
    size = "small",
    srcs = ["python/estimator/baseline_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":baseline",
        ":head",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:numpy_io",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["python/estimator/boosted_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:boosted_trees",
    ],
)

py_test(
    name = "boosted_trees_test",
    size = "medium",
    srcs = ["python/estimator/boosted_trees_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":boosted_trees",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:numpy_io",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dnn",
    srcs = ["python/estimator/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn",
    ],
)

py_test(
    name = "dnn_test",
    size = "medium",
    srcs = ["python/estimator/dnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":dnn",
        ":head",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:dnn_testing_utils",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["python/estimator/dnn_linear_combined.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
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
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:dnn_testing_utils",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:linear_testing_utils",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
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
        "//tensorflow:tensorflow_py_no_contrib",
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
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/contrib/predictor",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:linear",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "export",
    srcs = [
        "python/estimator/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_test(
    name = "export_test",
    size = "medium",
    srcs = ["python/estimator/export_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/62863147
    deps = [
        ":export",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_library(
    name = "exporter",
    srcs = [
        "python/estimator/exporter.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python/estimator:exporter",
    ],
)

py_test(
    name = "exporter_test",
    size = "medium",
    srcs = ["python/estimator/exporter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":exporter",
        "//tensorflow/python:platform",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:exporter",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["python/estimator/head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "hooks",
    srcs = [
        "python/estimator/hooks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_test(
    name = "hooks_test",
    size = "medium",
    srcs = ["python/estimator/hooks_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":hooks",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:estimator_py",
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
    size = "medium",
    srcs = ["python/estimator/linear_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":head",
        ":linear",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:linear_testing_utils",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:prediction_keys",
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
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:dnn",
        "//tensorflow/python/estimator:linear",
    ],
)

py_test(
    name = "logit_fns_test",
    size = "small",
    srcs = ["python/estimator/logit_fns_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logit_fns",
        "//tensorflow:tensorflow_py_no_contrib",
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
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
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
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:metric_keys",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:prediction_keys",
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
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:util",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "replicate_model_fn_test",
    size = "medium",
    srcs = ["python/estimator/replicate_model_fn_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:dnn",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:optimizers",
        "//tensorflow/python/estimator:prediction_keys",
        ":replicate_model_fn",
    ],
    tags = [
        "manual",
        "multi_gpu",
        "notap",
    ],
)

py_library(
    name = "rnn",
    srcs = ["python/estimator/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":extenders",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/contrib/feature_column:feature_column_py",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:optimizers",
        "@six_archive//:six",
    ],
)

py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["python/estimator/rnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # times out
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":head",
        ":rnn",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/contrib/data",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:parsing_utils",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "early_stopping",
    srcs = ["python/estimator/early_stopping.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
    ],
)

py_test(
    name = "early_stopping_test",
    srcs = ["python/estimator/early_stopping_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":early_stopping",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/estimator",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "saved_model_estimator",
    srcs = ["python/estimator/saved_model_estimator.py"],
    deps = [
        ":export",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:export",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/saved_model",
    ],
)

py_test(
    name = "saved_model_estimator_test",
    size = "medium",
    srcs = ["python/estimator/saved_model_estimator_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "notsan",
    ],
    deps = [
        ":export",
        ":saved_model_estimator",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:export_export",
        "//tensorflow/python/estimator:export_output",
        "//tensorflow/python/estimator:model_fn",
    ],
)
