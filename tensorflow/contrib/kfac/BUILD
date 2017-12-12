# Description:
#   Contains KfacOptimizer, an implementation of the K-FAC optimization
#   algorithm in TensorFlow.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "kfac",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/kfac/python/ops:curvature_matrix_vector_products_lib",
        "//tensorflow/contrib/kfac/python/ops:fisher_blocks_lib",
        "//tensorflow/contrib/kfac/python/ops:fisher_estimator_lib",
        "//tensorflow/contrib/kfac/python/ops:fisher_factors_lib",
        "//tensorflow/contrib/kfac/python/ops:kfac_optimizer_lib",
        "//tensorflow/contrib/kfac/python/ops:layer_collection_lib",
        "//tensorflow/contrib/kfac/python/ops:loss_functions_lib",
        "//tensorflow/contrib/kfac/python/ops:op_queue_lib",
        "//tensorflow/contrib/kfac/python/ops:utils_lib",
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
