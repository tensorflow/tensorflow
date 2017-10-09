package(default_visibility = [
    "//tensorflow/contrib/kfac:__pkg__",
    "//tensorflow/contrib/kfac/python/kernel_tests:__pkg__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "fisher_blocks",
    srcs = ["fisher_blocks.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fisher_factors",
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "@six_archive//:six",
    ],
)

py_library(
    name = "fisher_blocks_lib",
    srcs = ["fisher_blocks_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fisher_blocks",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "fisher_factors",
    srcs = ["fisher_factors.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "fisher_factors_lib",
    srcs = ["fisher_factors_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fisher_factors",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "loss_functions",
    srcs = ["loss_functions.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ops/distributions",
        "@six_archive//:six",
    ],
)

py_library(
    name = "loss_functions_lib",
    srcs = ["loss_functions_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss_functions",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "curvature_matrix_vector_products",
    srcs = ["curvature_matrix_vector_products.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "curvature_matrix_vector_products_lib",
    srcs = ["curvature_matrix_vector_products_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":curvature_matrix_vector_products",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "layer_collection",
    srcs = ["layer_collection.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fisher_blocks",
        ":loss_functions",
        ":utils",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "layer_collection_lib",
    srcs = ["layer_collection_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layer_collection",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "kfac_optimizer",
    srcs = [
        "optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":curvature_matrix_vector_products",
        ":fisher_estimator",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "kfac_optimizer_lib",
    srcs = [
        "optimizer_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":kfac_optimizer",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "fisher_estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "fisher_estimator_lib",
    srcs = [
        "estimator_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":fisher_estimator",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "utils_lib",
    srcs = ["utils_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "op_queue",
    srcs = ["op_queue.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "op_queue_lib",
    srcs = ["op_queue_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":op_queue",
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
