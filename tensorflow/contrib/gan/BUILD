# Files for using TFGAN framework.
package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "gan",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":eval",
        ":features",
        ":losses",
        ":namedtuples",
        ":train",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "namedtuples",
    srcs = ["python/namedtuples.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "train",
    srcs = ["python/train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
        ":namedtuples",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:learning",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
    ],
)

py_test(
    name = "train_test",
    srcs = ["python/train_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":namedtuples",
        ":train",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:learning",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "eval",
    srcs = ["python/eval/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":classifier_metrics",
        ":eval_utils",
        ":summaries",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "estimator",
    srcs = ["python/estimator/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gan_estimator",
        ":head",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "losses",
    srcs = ["python/losses/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        ":tuple_losses",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "features",
    srcs = ["python/features/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clip_weights",
        ":conditioning_utils",
        ":virtual_batchnorm",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "losses_impl",
    srcs = ["python/losses/python/losses_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "losses_impl_test",
    srcs = ["python/losses/python/losses_impl_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
    ],
)

py_library(
    name = "tuple_losses",
    srcs = [
        "python/losses/python/losses_wargs.py",
        "python/losses/python/tuple_losses.py",
        "python/losses/python/tuple_losses_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "tuple_losses_test",
    srcs = ["python/losses/python/tuple_losses_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tuple_losses",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "conditioning_utils",
    srcs = [
        "python/features/python/conditioning_utils.py",
        "python/features/python/conditioning_utils_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "conditioning_utils_test",
    srcs = ["python/features/python/conditioning_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":conditioning_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "virtual_batchnorm",
    srcs = [
        "python/features/python/virtual_batchnorm.py",
        "python/features/python/virtual_batchnorm_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "virtual_batchnorm_test",
    srcs = ["python/features/python/virtual_batchnorm_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":virtual_batchnorm",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clip_weights",
    srcs = [
        "python/features/python/clip_weights.py",
        "python/features/python/clip_weights_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/contrib/opt:opt_py"],
)

py_test(
    name = "clip_weights_test",
    srcs = ["python/features/python/clip_weights_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clip_weights",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "classifier_metrics",
    srcs = [
        "python/eval/python/classifier_metrics.py",
        "python/eval/python/classifier_metrics_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "classifier_metrics_test",
    srcs = ["python/eval/python/classifier_metrics_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":classifier_metrics",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "eval_utils",
    srcs = [
        "python/eval/python/eval_utils.py",
        "python/eval/python/eval_utils_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "eval_utils_test",
    srcs = ["python/eval/python/eval_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":eval_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "summaries",
    srcs = [
        "python/eval/python/summaries.py",
        "python/eval/python/summaries_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":eval_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/losses",
    ],
)

py_test(
    name = "summaries_test",
    srcs = ["python/eval/python/summaries_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":namedtuples",
        ":summaries",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/python/head.py",
        "python/estimator/python/head_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":namedtuples",
        ":train",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_test(
    name = "head_test",
    srcs = ["python/estimator/python/head_test.py"],
    shard_count = 1,
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        ":namedtuples",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_library(
    name = "gan_estimator",
    srcs = [
        "python/estimator/python/gan_estimator.py",
        "python/estimator/python/gan_estimator_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        ":namedtuples",
        ":summaries",
        ":train",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_test(
    name = "gan_estimator_test",
    srcs = ["python/estimator/python/gan_estimator_test.py"],
    shard_count = 1,
    srcs_version = "PY2AND3",
    deps = [
        ":gan_estimator",
        ":namedtuples",
        ":tuple_losses",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:numpy_io",
        "//third_party/py/numpy",
        "@six_archive//:six",
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
