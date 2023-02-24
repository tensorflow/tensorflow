package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "structured_function",
    srcs = ["structured_function.py"],
    srcs_version = "PY3",
    deps = [
        ":debug_mode",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "debug_mode",
    srcs = ["debug_mode.py"],
)

py_library(
    name = "test_mode",
    srcs = ["test_mode.py"],
)

py_library(
    name = "batch_op",
    srcs = ["batch_op.py"],
)

py_library(
    name = "counter_op",
    srcs = ["counter_op.py"],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "padded_batch_op",
    srcs = ["padded_batch_op.py"],
)

py_library(
    name = "take_op",
    srcs = ["take_op.py"],
)

py_library(
    name = "scan_op",
    srcs = ["scan_op.py"],
)

py_library(
    name = "shuffle_op",
    srcs = ["shuffle_op.py"],
)

py_library(
    name = "rebatch_op",
    srcs = ["rebatch_op.py"],
    deps = [
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "skip_op",
    srcs = ["skip_op.py"],
)

py_library(
    name = "dataset_ops",
    # Grouped together due to mutual dependencies, to avoid dependency cycles.
    srcs = [
        "dataset_autograph.py",
        "dataset_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":batch_op",
        ":cache_op",
        ":choose_from_datasets_op",
        ":concatenate_op",
        ":counter_op",
        ":debug_mode",
        ":directed_interleave_op",
        ":filter_op",
        ":flat_map_op",
        ":from_generator_op",
        ":from_sparse_tensor_slices_op",
        ":from_tensor_slices_op",
        ":from_tensors_op",
        ":group_by_window_op",
        ":ignore_errors_op",
        ":interleave_op",
        ":iterator_ops",
        ":load_ops",
        ":map_op",
        ":options",
        ":padded_batch_op",
        ":prefetch_op",
        ":ragged_batch_op",
        ":random_op",
        ":range_op",
        ":rebatch_op",
        ":repeat_op",
        ":sample_from_datasets_op",
        ":save_ops",
        ":scan_op",
        ":shard_op",
        ":shuffle_op",
        ":skip_op",
        ":snapshot_op",
        ":sparse_batch_op",
        ":structured_function",
        ":take_op",
        ":take_while_op",
        ":unbatch_op",
        ":unique_op",
        ":window_op",
        ":zip_op",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:smart_cond",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:options",
        "//tensorflow/python/data/util:random_seed",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/data/util:traverse",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/types",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cache_op",
    srcs = ["cache_op.py"],
)

py_library(
    name = "choose_from_datasets_op",
    srcs = ["choose_from_datasets_op.py"],
    deps = [
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "concatenate_op",
    srcs = ["concatenate_op.py"],
)

py_library(
    name = "directed_interleave_op",
    srcs = ["directed_interleave_op.py"],
)

py_library(
    name = "filter_op",
    srcs = ["filter_op.py"],
)

py_library(
    name = "flat_map_op",
    srcs = ["flat_map_op.py"],
)

py_library(
    name = "from_generator_op",
    srcs = ["from_generator_op.py"],
)

py_library(
    name = "from_sparse_tensor_slices_op",
    srcs = ["from_sparse_tensor_slices_op.py"],
)

py_library(
    name = "from_tensor_slices_op",
    srcs = ["from_tensor_slices_op.py"],
)

py_library(
    name = "from_tensors_op",
    srcs = ["from_tensors_op.py"],
)

py_library(
    name = "group_by_window_op",
    srcs = ["group_by_window_op.py"],
)

py_library(
    name = "random_op",
    srcs = ["random_op.py"],
)

py_library(
    name = "range_op",
    srcs = ["range_op.py"],
)

py_library(
    name = "repeat_op",
    srcs = ["repeat_op.py"],
)

py_library(
    name = "sample_from_datasets_op",
    srcs = ["sample_from_datasets_op.py"],
    deps = [
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "ignore_errors_op",
    srcs = ["ignore_errors_op.py"],
)

py_library(
    name = "interleave_op",
    srcs = ["interleave_op.py"],
)

py_library(
    name = "iterator_ops",
    # Grouped together due to mutual dependencies, to avoid dependency cycles.
    srcs = [
        "iterator_autograph.py",
        "iterator_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":optional_ops",
        ":options",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:type_utils",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/trackable:base",
    ],
)

py_library(
    name = "load_ops",
    srcs = ["load_op.py"],
    deps = [
        ":structured_function",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "map_op",
    srcs = ["map_op.py"],
)

py_library(
    name = "prefetch_op",
    srcs = ["prefetch_op.py"],
)

py_library(
    name = "ragged_batch_op",
    srcs = ["ragged_batch_op.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "sparse_batch_op",
    srcs = ["sparse_batch_op.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "save_ops",
    srcs = ["save_op.py"],
    deps = [
        ":structured_function",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "snapshot_op",
    srcs = ["snapshot_op.py"],
)

py_library(
    name = "take_while_op",
    srcs = ["take_while_op.py"],
)

py_library(
    name = "unique_op",
    srcs = ["unique_op.py"],
)

py_library(
    name = "unbatch_op",
    srcs = ["unbatch_op.py"],
)

py_library(
    name = "window_op",
    srcs = ["window_op.py"],
)

py_library(
    name = "zip_op",
    srcs = ["zip_op.py"],
    deps = [
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "multi_device_iterator_ops",
    srcs = ["multi_device_iterator_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset_ops",
        ":iterator_ops",
        ":options",
        ":prefetch_op",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:type_utils",
    ],
)

py_library(
    name = "optional_ops",
    srcs = ["optional_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:optional_ops_gen",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/saved_model:nested_structure_coder",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:test_mode",
        "//tensorflow/python/data/util:options",
    ],
)

py_library(
    name = "readers",
    srcs = ["readers.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/types",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "shard_op",
    srcs = ["shard_op.py"],
)
