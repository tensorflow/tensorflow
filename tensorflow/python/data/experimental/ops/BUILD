package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "batching",
    srcs = ["batching.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "cardinality",
    srcs = ["cardinality.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "compression_ops",
    srcs = ["compression_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
    ],
)

py_library(
    name = "counter",
    srcs = ["counter.py"],
    srcs_version = "PY3",
    deps = [
        ":scan_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "data_service_ops",
    srcs = [
        "data_service_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":compression_ops",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/service:_pywrap_server_lib",
        "//tensorflow/python/data/experimental/service:_pywrap_utils",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "distributed_save_op",
    srcs = [
        "distributed_save_op.py",
    ],
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/saved_model:nested_structure_coder",
    ],
)

py_library(
    name = "distribute",
    srcs = [
        "distribute.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/types",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "enumerate_ops",
    srcs = ["enumerate_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "error_ops",
    srcs = ["error_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "from_list",
    srcs = ["from_list.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "get_single_element",
    srcs = ["get_single_element.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/types",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "grouping",
    srcs = ["grouping.py"],
    srcs_version = "PY3",
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
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "interleave_ops",
    srcs = ["interleave_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":random_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "io",
    srcs = [
        "io.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "iterator_ops",
    srcs = [
        "iterator_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:basic_session_run_hooks",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:session_run_hook",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "lookup_ops",
    srcs = [
        "lookup_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cardinality",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "map_defun",
    srcs = ["map_defun.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "matching_files",
    srcs = ["matching_files.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["parsing_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "prefetching_ops",
    srcs = ["prefetching_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "random_access",
    srcs = ["random_access.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "readers",
    srcs = [
        "readers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":batching",
        ":error_ops",
        ":interleave_ops",
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "resampling",
    srcs = ["resampling.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "scan_ops",
    srcs = ["scan_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "shuffle_ops",
    srcs = [
        "shuffle_ops.py",
    ],
    srcs_version = "PY3",
    visibility = [
        # TODO(jsimsa): Remove non-default visibility exception when `index_shuffle` is made public.
        "//third_party/py/tensorflow_datasets:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":random_access",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "snapshot",
    srcs = [
        "snapshot.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "take_while_ops",
    srcs = ["take_while_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "testing",
    srcs = ["testing.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "unique",
    srcs = [
        "unique.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "writers",
    srcs = [
        "writers.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/types",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "dataset_ops",
    srcs_version = "PY3",
    deps = [
        ":batching",
        ":cardinality",
        ":compression_ops",
        ":counter",
        ":data_service_ops",
        ":distribute",
        ":distributed_save_op",
        ":enumerate_ops",
        ":error_ops",
        ":from_list",
        ":get_single_element",
        ":grouping",
        ":interleave_ops",
        ":io",
        ":map_defun",
        ":matching_files",
        ":prefetching_ops",
        ":random_access",
        ":readers",
        ":resampling",
        ":scan_ops",
        ":shuffle_ops",
        ":snapshot",
        ":take_while_ops",
        ":unique",
        ":writers",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:debug_mode",
        "//tensorflow/python/data/util:nest",
    ],
)
