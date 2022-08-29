package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "structured_function",
    srcs = ["structured_function.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "counter_op",
    srcs = ["counter_op.py"],
)

py_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":counter_op",
        ":from_tensor_slices_op",
        ":iterator_ops",
        ":load_ops",
        ":options",
        ":save_ops",
        ":structured_function",
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
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "from_tensor_slices_op",
    srcs = ["from_tensor_slices_op.py"],
)

py_library(
    name = "iterator_ops",
    srcs = ["iterator_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":optional_ops",
        ":options",
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
        "//tensorflow/python/util",
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
        "//tensorflow/python/util",
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
    ],
)

py_library(
    name = "optional_ops",
    srcs = ["optional_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:structure",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
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
        "//tensorflow/python/util:tf_export",
    ],
)
