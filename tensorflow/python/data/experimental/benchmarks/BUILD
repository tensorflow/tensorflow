package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "autotune_benchmark",
    srcs = ["autotune_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:optimization",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "csv_dataset_benchmark",
    srcs = ["csv_dataset_benchmark.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:readers",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "map_and_batch_benchmark",
    srcs = ["map_and_batch_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "map_defun_benchmark",
    srcs = ["map_defun_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/experimental/ops:map_defun",
        "//tensorflow/python/eager:function",
    ],
)

py_test(
    name = "map_vectorization_benchmark",
    srcs = ["map_vectorization_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "matching_files_benchmark",
    size = "small",
    srcs = ["matching_files_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:matching_files",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "choose_fastest_benchmark",
    srcs = ["choose_fastest_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "optimize_benchmark",
    srcs = ["optimize_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "parallel_interleave_benchmark",
    srcs = ["parallel_interleave_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/experimental/ops:optimization",
        "//tensorflow/python/data/experimental/ops:sleep",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "rejection_resample_benchmark",
    srcs = ["rejection_resample_benchmark.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:resampling",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "unbatch_benchmark",
    srcs = ["unbatch_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)
