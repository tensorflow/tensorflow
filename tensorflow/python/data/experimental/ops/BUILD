load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "batching",
    srcs = ["batching.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "cardinality",
    srcs = ["cardinality.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "compression_ops",
    srcs = ["compression_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "counter",
    srcs = ["counter.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "data_service_ops",
    srcs = [
        "data_service_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":compression_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/data/experimental/service:_pywrap_server_lib",
        "//tensorflow/python/data/experimental/service:_pywrap_utils_exp",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "distributed_save_op",
    srcs = [
        "distributed_save_op.py",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "distribute",
    srcs = [
        "distribute.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/types:data",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "enumerate_ops",
    srcs = ["enumerate_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "error_ops",
    srcs = ["error_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "from_list",
    srcs = ["from_list.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "get_single_element",
    srcs = ["get_single_element.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/types:data",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "global_shuffle_op",
    srcs = [
        "global_shuffle_op.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:random_seed",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "grouping",
    srcs = ["grouping.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "index_flat_map_op",
    srcs = [
        "index_flat_map_op.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "interleave_ops",
    srcs = ["interleave_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "io",
    srcs = [
        "io.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "iterator_ops",
    srcs = [
        "iterator_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/ops:proto_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "lookup_ops",
    srcs = [
        "lookup_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cardinality",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "map_defun",
    srcs = ["map_defun.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:dataset_ops_gen",
    ],
)

py_strict_library(
    name = "matching_files",
    srcs = ["matching_files.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "pad_to_cardinality",
    srcs = ["pad_to_cardinality.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "parsing_ops",
    srcs = ["parsing_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "prefetching_ops",
    srcs = ["prefetching_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "random_access",
    srcs = ["random_access.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "readers",
    srcs = [
        "readers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":error_ops",
        ":parsing_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "resampling",
    srcs = ["resampling.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "scan_ops",
    srcs = ["scan_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:random_seed",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "snapshot",
    srcs = [
        "snapshot.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "take_while_ops",
    srcs = ["take_while_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "testing",
    srcs = ["testing.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
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

py_strict_library(
    name = "weighted_flat_map_op",
    srcs = [
        "weighted_flat_map_op.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:random_seed",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "writers",
    srcs = [
        "writers.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/types:data",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
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
        ":global_shuffle_op",
        ":grouping",
        ":interleave_ops",
        ":io",
        ":map_defun",
        ":matching_files",
        ":pad_to_cardinality",
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:debug_mode",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/ops:dataset_ops_gen",
    ],
)
