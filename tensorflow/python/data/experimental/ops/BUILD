package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "cardinality",
    srcs = ["cardinality.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "counter",
    srcs = ["counter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":scan_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "get_single_element",
    srcs = ["get_single_element.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "iterator_ops",
    srcs = [
        "iterator_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:basic_session_run_hooks",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:session_run_hook",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
    ],
)

py_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "readers",
    srcs = [
        "readers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":batching",
        ":interleave_ops",
        ":optimization",
        ":parsing_ops",
        ":shuffle_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "shuffle_ops",
    srcs = [
        "shuffle_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "batching",
    srcs = ["batching.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":get_single_element",
        ":grouping",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/data/util:structure",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "distribute",
    srcs = [
        "distribute.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "enumerate_ops",
    srcs = ["enumerate_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "error_ops",
    srcs = ["error_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "grouping",
    srcs = ["grouping.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "interleave_ops",
    srcs = ["interleave_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":random_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:stateless_random_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "map_defun",
    srcs = ["map_defun.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "matching_files",
    srcs = ["matching_files.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "optimization",
    srcs = ["optimization.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "optimization_options",
    srcs = ["optimization_options.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:options",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["parsing_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "resampling",
    srcs = ["resampling.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":batching",
        ":interleave_ops",
        ":scan_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "scan_ops",
    srcs = ["scan_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "sleep",
    srcs = ["sleep.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "stats_aggregator",
    srcs = ["stats_aggregator.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "stats_ops",
    srcs = ["stats_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "stats_options",
    srcs = ["stats_options.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":stats_aggregator",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:options",
    ],
)

py_library(
    name = "take_while_ops",
    srcs = ["take_while_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "threading_options",
    srcs = ["threading_options.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:options",
    ],
)

py_library(
    name = "threadpool",
    srcs = ["threadpool.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "unique",
    srcs = [
        "unique.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "writers",
    srcs = [
        "writers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "indexed_dataset_ops",
    srcs = ["indexed_dataset_ops.py"],
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "prefetching_ops",
    srcs = ["prefetching_ops.py"],
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
    ],
)

py_library(
    name = "dataset_ops",
    deps = [
        ":batching",
        ":cardinality",
        ":counter",
        ":distribute",
        ":enumerate_ops",
        ":error_ops",
        ":get_single_element",
        ":grouping",
        ":indexed_dataset_ops",
        ":interleave_ops",
        ":map_defun",
        ":matching_files",
        ":optimization",
        ":prefetching_ops",
        ":readers",
        ":resampling",
        ":scan_ops",
        ":shuffle_ops",
        ":sleep",
        ":stats_ops",
        ":take_while_ops",
        ":threadpool",
        ":unique",
        ":writers",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)
